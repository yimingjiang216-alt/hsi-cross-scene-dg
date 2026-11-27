# -*- coding: utf-8 -*-
"""
Physics-Guided Domain Randomization and Spectral-Shape Invariant Representation
for Cross-Scene Hyperspectral Image Classification (Ours)

Three Core Innovations:
1. WavelengthPhysicalGenerator (Module 1):
   Simulates solar incidence scaling (alpha) + wavelength-dependent Rayleigh/Mie
   atmospheric scattering offset (beta * lambda^{-gamma}) + spatial-morphological style randomization.
2. SpectralShapeDisentangledEncoder (Module 2):
   Extracts 1st- and 2nd-order spectral derivatives (slopes & absorption valley curvatures)
   via `torch.diff` and fuses them with raw reflectance via a 1D Spectral Gating block.
3. Orthogonal Style-Content Disentanglement (Module 3):
   Explicitly separates domain-invariant material representation (z_inv) and
   domain-variant environmental style representation (z_env) with an orthogonal constraint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .generator import SpaRandomization, SpeRandomization, AdaIN2d
from .discriminator import MorphNet


class WavelengthPhysicalRandomization(nn.Module):
    """
    Innovation 1: Wavelength-Aware Physical Domain Randomization
    X_aug(c) = alpha * X(c) + beta * (lambda_c)^{-gamma} + eps
    """
    def __init__(self, num_bands, alpha_range=(0.75, 1.25), beta_range=(-0.15, 0.15), gamma_range=(0.5, 3.0)):
        super().__init__()
        self.num_bands = num_bands
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.gamma_range = gamma_range
        # Normalized wavelength coordinate in [0.2, 1.0] across C spectral bands
        lam = torch.linspace(0.2, 1.0, steps=num_bands).view(1, num_bands, 1, 1)
        self.register_buffer('wavelength_coord', lam)

    def forward(self, x):
        if not self.training:
            return x
        B, C, H, W = x.shape
        device = x.device
        # 1. Solar illumination & topographic shadow multiplicative factor alpha
        alpha = torch.empty(B, 1, 1, 1, device=device).uniform_(*self.alpha_range)
        # 2. Atmospheric Rayleigh/Mie scattering power-law exponent gamma and intensity beta
        gamma = torch.empty(B, 1, 1, 1, device=device).uniform_(*self.gamma_range)
        beta = torch.empty(B, 1, 1, 1, device=device).uniform_(*self.beta_range)
        scatter_curve = torch.pow(self.wavelength_coord, -gamma)
        scatter_curve = scatter_curve / (scatter_curve.mean(dim=1, keepdim=True) + 1e-6)
        # 3. Sensor thermal perturbation
        eps = torch.randn_like(x) * 0.015
        x_phys = alpha * x + beta * scatter_curve + eps
        return x_phys


class PhysicsGuidedGenerator(nn.Module):
    """
    Upgraded Domain Expansion Generator combining:
    - Physical wavelength-dependent atmospheric/illumination randomization
    - Spatial-Spectral-Morphological randomization
    """
    def __init__(self, n=64, kernelsize=3, imdim=48, imsize=[13, 13], zdim=10, device=0, use_phys_aug=True):
        super().__init__()
        stride = (kernelsize - 1) // 2
        self.zdim = zdim
        self.imdim = imdim
        self.imsize = imsize
        self.device = device
        self.use_phys_aug = use_phys_aug
        num_morph = 4

        self.phys_aug = WavelengthPhysicalRandomization(imdim)
        self.Morphology = MorphNet(imdim)
        self.adain2_morph = AdaIN2d(zdim, num_morph)

        self.conv_spa1 = nn.Conv2d(imdim, 3, 1, 1)
        self.conv_spa2 = nn.Conv2d(3, n, 1, 1)
        self.conv_spe1 = nn.Conv2d(imdim, n, imsize[0], 1)
        self.conv_spe2 = nn.ConvTranspose2d(n, n, imsize[0])
        self.conv1 = nn.Conv2d(n + n + num_morph, n, kernelsize, 1, stride)
        self.conv2 = nn.Conv2d(n, imdim, kernelsize, 1, stride)
        self.speRandom = SpeRandomization(n)
        self.spaRandom = SpaRandomization(3, device=device)

    def forward(self, x):
        if self.use_phys_aug:
            x_in = self.phys_aug(x)
        else:
            x_in = x

        x_morph = self.Morphology(x_in)
        z = torch.randn(len(x_in), self.zdim, device=x_in.device)
        x_morph = self.adain2_morph(x_morph, z)

        x_spa = F.relu(self.conv_spa1(x_in))
        x_spe = F.relu(self.conv_spe1(x_in))
        x_spa, idx_swap = self.spaRandom(x_spa)
        x_spe = self.speRandom(x_spe, idx_swap)
        x_spe = self.conv_spe2(x_spe)
        x_spa = self.conv_spa2(x_spa)

        out = F.relu(self.conv1(torch.cat((x_spa, x_spe, x_morph), 1)))
        out = torch.sigmoid(self.conv2(out))
        return out


class SpectralShapeGate(nn.Module):
    """
    1D Spectral-Shape Gating block scanning across continuous spectral channels
    to capture illumination-invariant absorption valley geometry.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1d_1 = nn.Conv1d(1, 8, kernel_size=5, padding=2, bias=False)
        self.conv1d_2 = nn.Conv1d(8, 1, kernel_size=5, padding=2, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: [B, C, H, W] -> pool spatial -> [B, 1, C]
        B, C, H, W = x.shape
        s = x.mean(dim=(-2, -1)).unsqueeze(1)  # [B, 1, C]
        s = F.gelu(self.conv1d_1(s))
        s = self.conv1d_2(s).squeeze(1)        # [B, C]
        gate = torch.sigmoid(self.norm(s)).view(B, C, 1, 1)
        return x * gate


class SpectralShapeDisentangledDiscriminator(nn.Module):
    """
    Upgraded Encoder & Classifier incorporating:
    - Innovation 2: First- and Second-Order Spectral Derivative Shape Stream (use_spec_diff=True)
    - Innovation 3: Dual Projection Heads for Invariant Content (z_inv) & Environment Style (z_env)
    """
    def __init__(self, inchannel, outchannel, num_classes, patch_size, use_spec_diff=True):
        super().__init__()
        dim = 512
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.use_spec_diff = use_spec_diff

        # Raw spectral-spatial stream
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)

        # Innovation 2: Spectral Derivative (Slope & Curvature) Shape Stream
        if self.use_spec_diff:
            # 1st derivative has (C-1) channels, padded to C; 2nd derivative padded to C
            self.shape_gate = SpectralShapeGate(inchannel)
            self.diff_conv1 = nn.Conv2d(inchannel * 2, 64, kernel_size=3, stride=1, padding=0)
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
            )

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        # Task classification head (driven solely by invariant representation)
        self.cls_head_src = nn.Linear(dim, num_classes)
        # Invariant material representation head z_inv
        self.pro_head = nn.Linear(dim, outchannel)
        # Innovation 3: Environmental style representation head z_env (for orthogonal disentanglement)
        self.env_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(dim // 2, outchannel),
        )

    def _compute_spectral_derivatives(self, x):
        # First-order spectral derivative (slope): dX/d(lambda)
        dx1 = torch.diff(x, n=1, dim=1)
        dx1 = F.pad(dx1, (0, 0, 0, 0, 0, 1), mode='replicate')
        # Second-order spectral derivative (absorption peak/valley curvature): d^2X/d(lambda)^2
        dx2 = torch.diff(x, n=2, dim=1)
        dx2 = F.pad(dx2, (0, 0, 0, 0, 1, 1), mode='replicate')
        return torch.cat([self.shape_gate(dx1), dx2], dim=1)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel, self.patch_size, self.patch_size))
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(1, -1)
        return out2.size(1)

    def forward(self, x, mode='test', return_env=False):
        in_size = x.size(0)
        feat_raw = self.relu1(self.conv1(x))

        if self.use_spec_diff:
            x_diff = self._compute_spectral_derivatives(x)
            feat_diff = F.relu(self.diff_conv1(x_diff), inplace=True)
            out1 = self.mp(self.fusion_gate(torch.cat([feat_raw, feat_diff], dim=1)) + feat_raw)
        else:
            out1 = self.mp(feat_raw)

        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            return self.cls_head_src(out4)
        elif mode == 'train':
            z_inv = F.normalize(self.pro_head(out4), dim=-1)
            clss = self.cls_head_src(out4)
            if return_env:
                z_env = F.normalize(self.env_head(out3), dim=-1)
                return clss, z_inv, z_env
            return clss, z_inv


class OrthogonalDisentangleLoss(nn.Module):
    """
    Innovation 3: Orthogonal Style-Content Disentanglement Loss
    Forces the cosine similarity between invariant material features z_inv
    and environmental style features z_env to be zero.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z_inv, z_env):
        # Both z_inv and z_env are L2-normalized: [B, D]
        cos_sim = torch.sum(z_inv * z_env, dim=-1).abs()
        return cos_sim.mean()
