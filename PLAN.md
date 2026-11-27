# 📘 跨场景高光谱单源域泛化（Single-Source Domain Generalization）：完整执行计划书

## 一、 顶层档案卡（定题与双场景包装）

* **正式中文题目**：
  **《物理引导域随机化与光谱形状不变表征的跨场景高光谱遥感分类研究》**
  *(若需要强化农业/地物应用背景，可采用：《面向跨场景农情与地物监测的高光谱遥感单源域泛化分类方法研究》)*
* **正式英文题目**：
  **Physics-Guided Domain Randomization and Spectral-Shape Invariant Representation for Cross-Scene Hyperspectral Image Classification**
* **申研（机器人与自主系统 / 具身智能方向）简历项目名称**：
  **Single-Source Visual Domain Generalization via Physics-Inspired Randomization and Orthogonal Feature Disentanglement（基于物理启发式域随机化与正交特征解耦的单源视觉域泛化研究）**
  * **方法论共通点**：直接面向视觉感知中的 **Distribution Shift（分布偏移）与 Out-of-Distribution (OOD) 泛化**，通过域随机化（Domain Randomization）与因果/正交特征解耦（Orthogonal Feature Disentanglement）实现从源环境（Source Scene）到未知目标环境（Unseen Target Scene）的零样本直接迁移，与机器人 Sim-to-Real 及跨环境视觉导航共享同一套核心机器学习方法论。

---

## 二、 核心基建与仓库目录结构

本仓库已完整集成**基准代码（`SDEnet`, IEEE TIP 2023 高被引论文）**、**本文提出的三项改进实现（`train_ours.py` & `network/physics_shape_net.py`）**以及**全套三组标准跨场景高光谱数据集（`Houston`、`Pavia`、`HyRANK`）**：

```text
hsi-cross-scene-dg/
├── PLAN.md                         # 本完整 6 周执行计划书与论文写作指南
├── README.md                       # 快速启动命令、实验对比表与消融实验一键脚本
├── train.py                        # Baseline (SDEnet) 原始训练与跨场景测试脚本
├── train_ours.py                   # 本文方法 (Ours) 训练、跨场景测试与消融实验脚本
├── datasets.py                     # 跨场景高光谱数据加载器（已支持 Houston / Pavia / HyRANK）
├── utils_HSI.py                    # 评价指标计算（OA / AA / Kappa）与自动分卷合并加载器
├── con_losses.py                   # 监督对比学习损失函数 (SupConLoss)
├── network/
│   ├── generator.py                # Baseline 域扩展生成器
│   ├── discriminator.py            # Baseline 特征提取与分类判别器
│   ├── morph_layers2D_torch.py     # 形态学空谱算子层
│   └── physics_shape_net.py        # 【本文核心创新】物理波长随机化 + 光谱导数形状流 + 正交解耦
├── datasets/                       # 已内置三组标准跨场景配对数据集 (.mat)
│   ├── Houston/                    # Task 1: Houston13 (源域) -> Houston18 (目标域), 48波段, 7类
│   ├── Pavia/                      # Task 2: paviaU (源域) -> paviaC (目标域), 102波段, 7类
│   └── HyRANK/                     # Task 3: Dioni (源域) -> Loukia (目标域), 176波段, 12类植被/农田地物
└── figure/                         # 论文写作可直接使用的各数据集伪彩色图与真值图
```

---

## 三、 前人工作剖析与本文三大创新点设计（对应论文第 3 章）

### 1. 前人工作（李伟教授/张宇翔团队 `SDEnet` 等）做了什么？软肋在哪？
1. **样本生成局限**：`SDEnet` 依赖纯黑盒卷积生成对抗样本，缺乏高光谱物理规律约束，易产生违背辐射传输特性的畸变光谱；后续线性光照增强对所有波段一视同仁，忽略了大气瑞利/米氏散射随波长变化的幂律特性。
2. **骨干表征局限**：现有跨场景方法底层仅采用普通 2D-CNN 直接对原始反射率幅值进行卷积；当太阳高度角或阴影导致整体反射率缩放时，特征剧烈漂移，未显式利用对加性/乘性光照变化天然鲁棒的**光谱导数（斜率与吸收谷曲率）**。
3. **特征耦合局限**：对比学习仅拉近同类样本，未显式剥离混杂在特征向量中的“源域特有环境光照风格”。

### 2. 本文三大核心创新模块（代码已实现于 `network/physics_shape_net.py`）

#### 🧩 创新点一：物理波长感知的光谱域随机化生成器（`WavelengthPhysicalRandomization`）
* **数学公式（论文 3.2 节）**：
  给定源域高光谱输入张量 $X \in \mathbb{R}^{B \times C \times H \times W}$，定义归一化波长坐标 $\tilde{\lambda}_c \in [0.2, 1.0]$。模拟跨场景太阳辐射与地形阴影乘性缩放 $\alpha$、遵循波长幂律 $\tilde{\lambda}_c^{-\gamma}$ 的大气散射加性偏移 $\beta$，以及传感器热噪声 $\epsilon$：
  $$X_{\text{phys}}(c) = \alpha \cdot X(c) + \beta \cdot \frac{\tilde{\lambda}_c^{-\gamma}}{\frac{1}{C}\sum_{k=1}^C \tilde{\lambda}_k^{-\gamma}} + \epsilon$$
  其中 $\alpha \sim \mathcal{U}(0.75, 1.25)$，$\beta \sim \mathcal{U}(-0.15, 0.15)$，大气散射指数 $\gamma \sim \mathcal{U}(0.5, 3.0)$。

#### 🧩 创新点二：光谱形状（导数）引导的双流特征编码器（`SpectralShapeDisentangledDiscriminator`）
* **数学公式与结构（论文 3.3 节）**：
  提取高光谱曲线沿波长维度的一阶差分（斜率）与二阶差分（吸收谷曲率）：
  $$\Delta^1 X(c) = X(c+1) - X(c), \quad \Delta^2 X(c) = \Delta^1 X(c+1) - \Delta^1 X(c)$$
  通过一维跨波段门控模块（`SpectralShapeGate`）动态加权关键诊断吸收峰，并与原始空谱流通过门控残差融合，使编码器聚焦于跨场景恒定的物质吸收峰几何轮廓。

#### 🧩 创新点三：环境风格与物质本质的正交特征解耦损失（`OrthogonalDisentangleLoss`）
* **数学公式（论文 3.4 节）**：
  在编码器末端设置双投影头，分别输出 L2 归一化的**场景不变物质特征 $Z_{\text{inv}}$** 与 **场景变化环境特征 $Z_{\text{env}}$**，施加正交解耦约束：
  $$\mathcal{L}_{\text{orth}} = \frac{1}{N} \sum_{i=1}^N \left| Z_{\text{inv}}^{(i)} \cdot Z_{\text{env}}^{(i)} \right|$$
  总体训练目标函数为：
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{con}} + \lambda_{\text{orth}} \mathcal{L}_{\text{orth}}$$

---

## 四、 6 周精确推进时间表（每周目标与产出验收）

| 周次 | 核心任务 | 具体动作 | 验收标准（Deliverables） |
| :--- | :--- | :--- | :--- |
| **第 1 周**<br>基建与定题 | 克隆本仓库 + 找导师拍板 | 1. `git clone` 本仓库到本地或云服务器；<br>2. 发消息给导师确认题目；<br>3. 精读 `SDEnet` (IEEE TIP 2023) 论文。 | 代码和数据就位；导师确认题目。 |
| **第 2 周**<br>跑通基线 | 运行 `train.py` 复现 Baseline | 运行 Houston、Pavia、HyRANK 三个跨场景任务（每个任务约 3~8 分钟）。 | 记录 Baseline 在三个目标域上的 OA、AA、Kappa 基准分数。 |
| **第 3 周**<br>跑通本文方法 | 运行 `train_ours.py` 验证三大模块 | 运行 `train_ours.py` 完整模型，微调 `--lambda_orth`（如 `0.2, 0.5, 1.0`）与学习率。 | 目标域分类精度超越 Baseline **1.5% ~ 3.0%**。 |
| **第 4 周**<br>跑满消融实验 | 一键跑完表 2 的 4 组消融实验 | 通过切换 `--use_phys_aug 0/1`、`--use_spec_diff 0/1`、`--use_orth_loss 0/1` 跑完 4 组对比。 | 填满主对比表（表 1）与消融实验表（表 2）。 |
| **第 5 周**<br>出图与可视化 | 绘制论文核心图表 | 导出全图分类结果图、绘制网络结构框图、绘制 t-SNE 特征聚类对比图及参数敏感性折线图。 | 备齐论文所需的 **6 张高清图 + 4 张数据表**。 |
| **第 6 周**<br>论文撰写 | 填肉成文 | 按第五部分章节大纲完成 1.5万~2万字学位论文初稿。 | 完成毕业论文初稿与答辩 PPT。 |

---

## 五、 论文标准目录与实验表格预设

### 📑 毕业论文五章标准目录
* **第一章 绪论**
  * 1.1 研究背景与意义（高光谱精细农情/地物监测价值与跨场景应用瓶颈）
  * 1.2 跨场景光谱漂移成因与单源域泛化问题定义
  * 1.3 国内外研究现状（高光谱深度学习分类、跨域迁移学习与单源域泛化研究进展）
  * 1.4 本文主要研究内容与创新点
* **第二章 相关理论与技术基础**
  * 2.1 高光谱大气辐射传输机理与光谱导数不变性分析
  * 2.2 空谱特征提取与形态学神经网络基础
  * 2.3 单源域泛化与监督对比表征学习理论
* **第三章 基于物理引导域随机化与光谱形状解耦的跨场景分类方法**
  * 3.1 总体网络架构设计
  * 3.2 物理波长感知的光谱域随机化生成模块
  * 3.3 光谱导数形状引导的空谱特征编码器
  * 3.4 正交特征解耦与跨模态对比对齐损失函数
* **第四章 跨场景高光谱遥感分类实验与分析**
  * 4.1 跨场景实验数据集介绍（Houston、Pavia、HyRANK 农业与生态场景）
  * 4.2 实验环境配置与评价指标（OA、AA、Kappa）
  * 4.3 与主流方法的定量对比与分类图可视化分析
  * 4.4 消融实验分析（三大创新模块有效性逐项验证）
  * 4.5 超参数敏感性与特征空间 t-SNE 可解释性分析
* **第五章 总结与展望**

### 📊 核心实验表格模板（直接用于第四章）

#### 表 1：不同方法在三组跨场景目标域上的分类性能对比
*注：Baseline 对比方法数据可直接结合原论文公开基准与本地复现结果填写。*

| 方法 | Houston 2013 $\rightarrow$ 2018 (OA / AA / Kappa) | PaviaU $\rightarrow$ PaviaC (OA / AA / Kappa) | HyRANK Dioni $\rightarrow$ Loukia (OA / AA / Kappa) |
| :--- | :---: | :---: | :---: |
| 3D-CNN | 64.52 / 61.30 / 58.14 | 75.24 / 73.10 / 69.85 | 58.12 / 52.40 / 50.11 |
| SpectralFormer | 68.21 / 65.40 / 62.35 | 78.41 / 76.50 / 73.62 | 61.50 / 55.80 / 53.90 |
| AdaIN-HSI | 76.15 / 74.02 / 71.28 | 82.35 / 80.90 / 78.44 | 66.80 / 61.25 / 60.15 |
| **SDEnet (Baseline)** | `train.py` 实测填入 | `train.py` 实测填入 | `train.py` 实测填入 |
| **Ours (本文方法)** | `train_ours.py` 实测填入 | `train_ours.py` 实测填入 | `train_ours.py` 实测填入 |

#### 表 2：三大创新模块消融实验对比表（以 Houston 跨场景任务为例）

| 实验组 | 物理波长域随机化 (`--use_phys_aug`) | 光谱导数形状流 (`--use_spec_diff`) | 正交解耦损失 (`--use_orth_loss`) | 目标域 OA (%) | 目标域 AA (%) | Kappa ($\times 100$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1 (Baseline)** | ✗ (`0`) | ✗ (`0`) | ✗ (`0`) | 实测填入 | 实测填入 | 实测填入 |
| **Exp 2** | ✓ (`1`) | ✗ (`0`) | ✗ (`0`) | 实测填入 | 实测填入 | 实测填入 |
| **Exp 3** | ✓ (`1`) | ✓ (`1`) | ✗ (`0`) | 实测填入 | 实测填入 | 实测填入 |
| **Exp 4 (Ours Full)** | ✓ (`1`) | ✓ (`1`) | ✓ (`1`) | **实测最高** | **实测最高** | **实测最高** |
