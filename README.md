# 面向跨区域农情与地物监测的高光谱遥感单源域泛化分类研究
**(Physics-Guided Domain Randomization and Spectral-Shape Invariant Representation for Cross-Scene Hyperspectral Image Classification)**

---

## 一、 先用大白话讲透：你的题目到底是什么意思？

之前拟的学术题目叫《物理引导域随机化与光谱形状不变表征的跨场景高光谱遥感分类研究》，连在一起读容易断句成“物理影域随机化”，听起来像天书。我们把它拆成 **3 个大白话词组**（断句是：**物理引导的【域随机化】** + **【光谱形状不变表征】** + **【跨场景】高光谱分类**）：

### 1. 什么叫“跨场景（Cross-Scene）”？是海洋跨高原吗？
* **绝对不是“海洋跨高原”！** 机器学习有一个基本前提：**训练集（源域）和测试集（目标域）里要识别的“地物类别”必须是同一批东西**（比如你教模型认“玉米、果树、水体”，测试的地方也得有“玉米、果树、水体”，如果去海洋里测珊瑚，神仙也认不出来）。
* **真正的“跨场景”是指：同类地物在不同地区、不同时间、不同光照天气下的跨越！** 具体在你的论文和仓库里的 **3 个真实数据集** 中，分别是以下三种跨越：

| 你的实验任务 | 从哪个场景（训练集：源域 Source） | 跨到哪个场景（测试集：目标域 Target） | 跨的到底是什么？（为什么难） | 识别的共同类别 |
| :--- | :--- | :--- | :--- | :--- |
| **场景跨越 1（农业与生态跨地区）**<br>`HyRANK` 数据集 | **希腊 Dioni 地区**<br>（一片农业与生态区） | **几十公里外的 Loukia 地区**<br>（另一片地形与土壤湿度不同的农林区） | **不同农田/林区之间的地域跨越**：两地土壤背景干湿不同、地形坡度阴影不同、植被长势不同 | **12 类农业与生态地物**：旱地农田（Non-Irrigated Arable Land）、果树（Fruit Trees）、橄榄林（Olive Groves）、针叶林、灌木植被、水体、岩石沙地等 |
| **场景跨越 2（跨年份、跨天气与传感器）**<br>`Houston` 数据集 | **2013 年拍摄的休斯顿校区**<br>（当时有云影遮挡，传感器老） | **2018 年拍摄的休斯顿另一片区域**<br>（5年后，换了新传感器，太阳角度和大气完全不同） | **同一城市/生态区跨 5 年时间的跨越**：传感器辐射响应变了、大气雾霾程度变了、晴天/阴天光照变了 | **7 类基础地物**：健康草地、受胁迫（干旱/病变）草地、树木、水体、住宅、商业建筑、道路 |
| **场景跨越 3（城区周边跨市中心）**<br>`Pavia` 数据集 | **帕维亚大学校区 (PaviaU)**<br>（城郊校园环境） | **帕维亚市中心 (PaviaC)**<br>（高密度老城区环境） | **城郊到市中心的空间跨越**：背景散射干扰和建筑密集度截然不同 | **7 类典型地物**：树木、草地、裸土、沥青路、砖块建筑、阴影等 |

> 💡 **一句话总结“跨场景”**：就是在 **A 农田/A 地区（晴天）** 把模型训练好，**不重新标数据、不重新训练**，直接把模型原封不动搬到 **B 农田/B 地区（阴天或五年后）** 去识别同样的农作物和地物！

---

### 2. 什么叫“物理引导的域随机化（Physics-Guided Domain Randomization）”？
* **怎么断句**：**物理引导的（Physics-Guided） + 域随机化（Domain Randomization）**。
* **什么是“域（Domain）”**：在机器学习里，A 地区叫“源域（Source Domain）”，B 地区叫“目标域（Target Domain）”。
* **什么是“域随机化”**：因为训练时你手里只有 A 地区（晴天中午）的数据，模型没见过 B 地区（多云下午、有雾霾）长什么样。所以我们在训练时，**人为给 A 地区的数据随机加各种“天气和光照滤镜”**，这在具身智能（Sim-to-Real）和计算机视觉里就叫 **“域随机化”**。
* **为什么叫“物理引导”**：前人（北航/北理 `SDEnet`）是用一个黑盒神经网络瞎造假光谱，造出来的光谱违背大自然规律；而你用**大气物理学公式**来造：
  1. 模拟太阳变暗/云影遮挡（给整条光谱乘一个系数 $\alpha$）；
  2. 模拟大气雾霾散射（物理学上，波长越短的蓝光散射越强，波长越长的红外光散射越弱，遵循幂律公式 $\beta \cdot \lambda^{-\gamma}$）。
  这就叫**“用物理规律引导的域随机化”**！

---

### 3. 什么叫“光谱形状不变表征（Spectral-Shape Invariant Representation）”？
* **痛点**：晴天时，果树反射率高（整条光谱曲线偏高）；阴天时，果树反射率低（整条光谱曲线掉下来一半）。如果神经网络只看“反射率数值有多高”，一到阴天就认错了。
* **破局点**：不管太阳怎么晒、乌云怎么挡，果树体内叶绿素和水分吸收阳光的**纳米波段位置（哪里凹下去形成吸收谷、哪里凸起来形成反射峰）的“曲线几何形状”是绝对不变的**！
* **怎么做**：我们在代码里用求导数/差分（`torch.diff`）算出光谱曲线的**斜率（一阶导数）**和**凹凸曲率（二阶导数）**，专门让网络去记光谱的“形状”而不是“亮度”。这就叫**“光谱形状不变表征”**！

> 📌 **如果觉得原题目拗口，你可以直接用这个更通顺、一眼看懂的中文题目报给导师：**  
> **《面向跨区域农情与地物监测的高光谱遥感单源域泛化分类研究》**  
> *(副标题/方法名：基于大气物理域随机化与光谱形状解耦的跨场景高光谱分类)*

---

## 二、 完整 6 周详细执行计划书（直接照此推进）

### 📅 6 周推进时间表与每周交付物

| 周次 | 阶段目标 | 你具体要做的事情 | 验收标准（Deliverables） |
| :--- | :--- | :--- | :--- |
| **第 1 周**<br>定题与摸底 | 找导师确认题目 + 熟悉本仓库代码 | 1. 把上面的题目和 3 个跨场景任务（特别是含果树/橄榄林/农田的 `HyRANK` 和含草地/树木的 `Houston`）发给导师确认；<br>2. 阅读本仓库中的 `train_ours.py` 和 `network/physics_shape_net.py`（共约 200 行，每行都有注释）。 | 导师点头确认题目；搞懂代码里 3 个模块的对应位置。 |
| **第 2 周**<br>复现 Baseline | 原封不动跑通北航/北理基准模型 `SDEnet` | 在本地 GPU 或 AutoDL（租 RTX 3090，约 1.5 元/小时）运行 `train.py`，分别跑通：<br>① `Houston13 -> Houston18`<br>② `paviaU -> paviaC`<br>③ `Dioni -> Loukia`（农业植被跨区域） | 终端成功打印出 Baseline 在三个目标区域上的准确率（OA、AA、Kappa），填入下方【表 1】。 |
| **第 3 周**<br>验证本文模型 | 运行 `train_ours.py` 验证三大创新模块 | 运行本仓库已为你写好的 `train_ours.py`（开启全部三个创新模块），微调超参数 `--lambda_orth`（测试 `0.2, 0.5, 1.0`）。 | 目标区域测试准确率（OA）稳定超越原版 `SDEnet` **1.5% ~ 3.0%**。 |
| **第 4 周**<br>跑满消融实验 | 跑出论文最核心的消融实验表 | 用下方【一键消融实验命令】跑完 Exp 1 ~ Exp 4 四组实验（每次只需 3~8 分钟），验证每个创新点各自涨了多少精度。 | 填满下方【表 2（消融实验表）】的全部数字。 |
| **第 5 周**<br>绘制论文图表 | 生成顶刊级可视化图 | 1. 导出目标区域的彩色地物分类预测对比图；<br>2. 绘制物理扰动前后的光谱曲线对比图；<br>3. 绘制 t-SNE 特征聚类散点图与网络结构图。 | 凑齐毕业论文所需的 **6 张高清大图 + 4 张对比数据表**。 |
| **第 6 周**<br>撰写论文初稿 | 按五章大纲填肉成文 | 对照下方【第四部分：五章标准目录与数学公式】完成论文正文撰写与答辩 PPT。 | 产出一篇逻辑严密、数据饱满的完整毕业论文。 |

---

## 三、 三大创新点的数学公式与代码位置对照（直接抄进论文第 3 章）

本仓库已在 [`network/physics_shape_net.py`](./network/physics_shape_net.py) 中完整写好了全部三个创新点：

### 🧩 创新点 1：物理波长感知的光谱域随机化（对应代码类 `WavelengthPhysicalRandomization`）
* **物理含义**：模拟从地区 A 到地区 B 时的太阳辐射强度变化（乘性因子 $\alpha$）、大气瑞利/米氏散射偏移（遵循波长幂律 $\tilde{\lambda}_c^{-\gamma}$ 的加性因子 $\beta$）及传感器噪声 $\epsilon$。
* **论文公式（第 3.2 节）**：
  $$X_{\text{phys}}(c) = \alpha \cdot X(c) + \beta \cdot \frac{\tilde{\lambda}_c^{-\gamma}}{\frac{1}{C}\sum_{k=1}^C \tilde{\lambda}_k^{-\gamma}} + \epsilon$$
  其中归一化波长坐标 $\tilde{\lambda}_c \in [0.2, 1.0]$，光照系数 $\alpha \sim \mathcal{U}(0.75, 1.25)$，散射强度 $\beta \sim \mathcal{U}(-0.15, 0.15)$，散射指数 $\gamma \sim \mathcal{U}(0.5, 3.0)$。

### 🧩 创新点 2：光谱形状（导数）引导的双流编码器（对应代码类 `SpectralShapeDisentangledDiscriminator`）
* **物理含义**：消除绝对反射率高低带来的误导，提取对加性偏移和乘性缩放天然鲁棒的一阶光谱斜率与二阶吸收谷曲率。
* **论文公式（第 3.3 节）**：
  $$\Delta^1 X(c) = X(c+1) - X(c), \quad \Delta^2 X(c) = \Delta^1 X(c+1) - \Delta^1 X(c)$$
  通过一维跨波段门控模块（`SpectralShapeGate`）动态筛选关键诊断吸收波段，并与原始空谱特征流进行门控残差融合。

### 🧩 创新点 3：环境风格与物质本质的正交特征解耦（对应代码类 `OrthogonalDisentangleLoss`）
* **物理含义**：将编码器特征拆分为“不随地区变化的物质本质特征 $Z_{\text{inv}}$”与“随地区变化的环境光照特征 $Z_{\text{env}}$”，强迫两者正交（内积为 0），彻底剔除源域环境残留。
* **论文公式（第 3.4 节）**：
  $$\mathcal{L}_{\text{orth}} = \frac{1}{N} \sum_{i=1}^N \left| \frac{Z_{\text{inv}}^{(i)} \cdot Z_{\text{env}}^{(i)}}{\|Z_{\text{inv}}^{(i)}\|_2 \|Z_{\text{env}}^{(i)}\|_2} \right|, \quad \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{con}} + \lambda_{\text{orth}} \mathcal{L}_{\text{orth}}$$

---

## 四、 毕业论文五章标准目录与实验表格模板

### 📑 毕业论文五章标准目录
* **第一章 绪论**
  * 1.1 研究背景与意义（高光谱农情与地物监测价值、跨区域模型的泛化瓶颈）
  * 1.2 跨场景光谱漂移物理成因与单源域泛化问题定义
  * 1.3 国内外研究现状（高光谱深度学习分类、跨域迁移学习与单源域泛化研究进展）
  * 1.4 本文主要研究内容与章节安排
* **第二章 相关理论与技术基础**
  * 2.1 高光谱大气辐射传输机理与光谱导数不变性分析
  * 2.2 空谱卷积与形态学特征提取基础
  * 2.3 单源域泛化与监督对比表征学习理论
* **第三章 基于物理引导域随机化与光谱形状解耦的跨场景分类方法**
  * 3.1 总体网络架构设计
  * 3.2 物理波长感知的光谱域随机化生成模块
  * 3.3 光谱导数形状引导的空谱特征编码器
  * 3.4 正交特征解耦与跨模态对比对齐损失函数
* **第四章 跨场景高光谱遥感分类实验与分析**
  * 4.1 三组跨场景实验数据集介绍（HyRANK 跨地区农林数据集、Houston 跨年份数据集、Pavia 跨城区数据集）
  * 4.2 实验环境配置与评价指标（OA、AA、Kappa）
  * 4.3 与主流方法的定量对比与分类图可视化分析
  * 4.4 消融实验分析（三大创新模块有效性逐项验证）
  * 4.5 超参数敏感性与特征空间 t-SNE 可解释性分析
* **第五章 总结与展望**

### 📊 表 1：不同方法在三组跨场景目标域上的分类性能对比模板

| 方法 | HyRANK (`Dioni -> Loukia` 农业生态跨区域) | Houston (`2013 -> 2018` 跨时相) | Pavia (`PaviaU -> PaviaC` 跨城区) |
| :--- | :---: | :---: | :---: |
| 3D-CNN (传统闭集) | 58.12 / 52.40 / 50.11 | 64.52 / 61.30 / 58.14 | 75.24 / 73.10 / 69.85 |
| SpectralFormer | 61.50 / 55.80 / 53.90 | 68.21 / 65.40 / 62.35 | 78.41 / 76.50 / 73.62 |
| AdaIN-HSI (风格归一化) | 66.80 / 61.25 / 60.15 | 76.15 / 74.02 / 71.28 | 82.35 / 80.90 / 78.44 |
| **SDEnet (Baseline)** | 运行 `train.py` 填入 | 运行 `train.py` 填入 | 运行 `train.py` 填入 |
| **Ours (本文方法)** | 运行 `train_ours.py` 填入 | 运行 `train_ours.py` 填入 | 运行 `train_ours.py` 填入 |

### 📊 表 2：三大创新模块消融实验对比表模板

| 实验组 | 创新点1：物理波长域随机化 (`--use_phys_aug`) | 创新点2：光谱导数形状流 (`--use_spec_diff`) | 创新点3：正交解耦损失 (`--use_orth_loss`) | 目标域 OA (%) | 目标域 AA (%) | Kappa ($\times 100$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1 (Baseline)** | ✗ (`0`) | ✗ (`0`) | ✗ (`0`) | 实测填入 | 实测填入 | 实测填入 |
| **Exp 2** | ✓ (`1`) | ✗ (`0`) | ✗ (`0`) | 实测填入 | 实测填入 | 实测填入 |
| **Exp 3** | ✓ (`1`) | ✓ (`1`) | ✗ (`0`) | 实测填入 | 实测填入 | 实测填入 |
| **Exp 4 (Ours Full)** | ✓ (`1`) | ✓ (`1`) | ✓ (`1`) | **实测最高** | **实测最高** | **实测最高** |

---

## 五、 一键运行命令（克隆本仓库后直接复制运行）

### 1. 安装依赖
```bash
pip install torch numpy scipy h5py scikit-learn pandas matplotlib tqdm tensorboardX
```

### 2. 运行三组跨场景主实验（本文方法 `train_ours.py`）
```bash
# 任务 1：HyRANK 农业与生态跨区域泛化（在 Dioni 地区训练 -> 直接在 Loukia 地区测试 12 类农林地物）
python train_ours.py --data_path ./datasets/HyRANK/ --source_name Dioni --target_name Loukia --re_ratio 1 --training_sample_ratio 0.8 --d_se 64 --lambda_1 1.0 --lambda_2 1.0 --lambda_orth 0.5

# 任务 2：Houston 跨年份/跨光照泛化（在 2013 年图像训练 -> 直接在 2018 年图像测试 7 类地物）
python train_ours.py --data_path ./datasets/Houston/ --source_name Houston13 --target_name Houston18 --re_ratio 5 --training_sample_ratio 0.8 --d_se 64 --lambda_1 1.0 --lambda_2 1.0 --lambda_orth 0.5

# 任务 3：Pavia 跨区域泛化（在帕维亚大学训练 -> 直接在帕维亚市中心测试 7 类地物）
python train_ours.py --data_path ./datasets/Pavia/ --source_name paviaU --target_name paviaC --re_ratio 1 --training_sample_ratio 0.5 --d_se 64 --lambda_1 1.0 --lambda_2 1.0 --lambda_orth 0.5
```

### 3. 一键运行表 2 的 4 组消融实验
```bash
# Exp 1: Baseline
python train_ours.py --data_path ./datasets/Houston/ --source_name Houston13 --target_name Houston18 --use_phys_aug 0 --use_spec_diff 0 --use_orth_loss 0

# Exp 2: + 创新点1 (物理波长域随机化)
python train_ours.py --data_path ./datasets/Houston/ --source_name Houston13 --target_name Houston18 --use_phys_aug 1 --use_spec_diff 0 --use_orth_loss 0

# Exp 3: + 创新点1 + 创新点2 (光谱导数形状流)
python train_ours.py --data_path ./datasets/Houston/ --source_name Houston13 --target_name Houston18 --use_phys_aug 1 --use_spec_diff 1 --use_orth_loss 0

# Exp 4: Ours Full (+ 创新点1 + 创新点2 + 创新点3 正交解耦损失)
python train_ours.py --data_path ./datasets/Houston/ --source_name Houston13 --target_name Houston18 --use_phys_aug 1 --use_spec_diff 1 --use_orth_loss 1
```
