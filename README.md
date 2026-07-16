# Deep Learning Coursework

> 深度学习课程作业合集 | BSAI @ M.U.S.T.

三门核心作业，从底层实现到前沿模型，覆盖现代深度学习的完整知识链。

---

## 目录

```
DL/
├── DL-HW1/     # 深度学习基础：从零实现
├── DL-HW2/     # 现代架构与生成模型
├── DL-HW3/     # 前沿专题
└── mnist_compare.py   # MNIST 分类器对比实验
```

---

## DL-HW1 — 深度学习基础

从 NumPy 纯手写实现开始，逐步过渡到 PyTorch。

| Notebook | 内容 | 关键技术 |
|----------|------|----------|
| `linear_classifier.ipynb` | 线性分类器 (SVM + Softmax) | 梯度计算、向量化损失、超参数搜索 |
| `two_layer_net.ipynb` | 两层全连接网络 | 反向传播推导、SGD 优化、调参 |
| `fully_connected_networks.ipynb` | 模块化全连接网络 | PyTorch 等价实现、BatchNorm、Dropout |
| `convolutional_networks.ipynb` | 卷积神经网络 | Conv2d、MaxPool、BatchNorm、过拟合分析 |

**数据集**：CIFAR-10、MNIST

---

## DL-HW2 — 现代架构与生成模型

深度学习前沿架构的核心原理与动手实现。

| Notebook | 内容 | 关键技术 |
|----------|------|----------|
| `11_1_Shattered_Gradients` | 梯度破碎问题 | 深度网络的梯度退化分析 |
| `11_2_Residual_Networks` | 残差网络 | Skip Connection、恒等映射、梯度流动 |
| `11_3_Batch_Normalization` | 批归一化 | 训练/推理模式差异、可学习参数 |
| `12_1_Self_Attention` | 自注意力机制 | QKV 计算、注意力权重可视化 |
| `12_2_Multihead_Self_Attention` | 多头注意力 | Head 划分与合并、并行注意力 |
| `12_3_Tokenization` | Tokenization | 文本到向量的完整流程 |
| `13_1_Graph_Representation` | 图表示学习 | 邻接矩阵、节点特征 |
| `13_2_Graph_Classification` | 图分类 | GNN 消息传递、图级别池化 |
| `13_3_Neighborhood_Sampling` | 邻域采样 | 大规模图训练的采样策略 |
| `15_1_GAN_Toy_Example` | 生成对抗网络 | Generator vs Discriminator 训练博弈 |
| `15_2_Wasserstein_Distance` | WGAN | Wasserstein 距离、Lipschitz 约束 |

---

## DL-HW3 — 前沿专题

生成模型、强化学习、训练动力学、可信AI 四大板块。

### 生成模型

| Notebook | 内容 |
|----------|------|
| `16_1_1D_Normalizing_Flows` | 标准流：从简单分布变换到复杂分布 |
| `16_2_Autoregressive_Flows` | 自回归流：条件分布建模 |
| `16_3_Contraction_Mappings` | 收缩映射：Banach 不动点定理 |
| `17_1_Latent_Variable_Models` | 隐变量模型：编码-解码框架 |
| `17_2_Reparameterization_Trick` | 重参数化技巧：让 VAE 可训练 |
| `17_3_Importance_Sampling` | 重要性采样：ELBO 与 tighter bound |
| `18_1_Diffusion_Encoder` | 扩散编码器：正向加噪过程 |
| `18_2_1D_Diffusion_Model` | 一维扩散模型：逆向去噪训练 |
| `18_3_Reparameterized_Model` | 重参数化扩散：噪声预测 vs 数据预测 |
| `18_4_Families_of_Diffusion_Models` | 扩散模型家族：DDPM、SDE 统一视角 |

### 强化学习

| Notebook | 内容 |
|----------|------|
| `19_1_Markov_Decision_Processes` | 马尔可夫决策过程形式化 |
| `19_2_Dynamic_Programming` | 动态规划：策略迭代与值迭代 |
| `19_3_Monte_Carlo_Methods` | 蒙特卡洛方法：首次访问 vs 每次访问 |
| `19_4_Temporal_Difference_Methods` | 时序差分：SARSA、Q-Learning |
| `19_5_Control_Variates` | 控制变量法：降低方差估计 |

### 训练动力学

| Notebook | 内容 |
|----------|------|
| `20_1_Random_Data` | 随机标签下的泛化行为 |
| `20_2_Full_Batch_Gradient_Descent` | 全批量梯度下降的收敛分析 |
| `20_3_Lottery_Tickets` | 彩票假说：稀疏子网络的发现 |
| `20_4_Adversarial_Attacks` | 对抗攻击：FGSM、PGD |

### 可信 AI

| Notebook | 内容 |
|----------|------|
| `21_1_Bias_Mitigation` | 模型偏差缓解 |
| `21_2_Explainability` | 模型可解释性：Saliency Map |

---

## 补充实验

`mnist_compare.py` — 在同一数据集上对比多种分类器（SVM、Softmax、MLP、CNN），分析参数量-精度-训练速度的 trade-off。

---

## 学习轨迹

```
线性分类器 → 全连接网络 → CNN
    │
    ├─ ResNet → BatchNorm → Attention → GNN
    │       └─ GAN → WGAN
    │
    └─ Normalizing Flows → VAE → Diffusion Models
        └─ MDP → DP → MC → TD → RL
            └─ Lottery Ticket → Adversarial → Explainability
```

---

## 运行环境

- Python 3.10+
- PyTorch
- NumPy, Matplotlib
- Jupyter Notebook / JupyterLab

```bash
pip install torch numpy matplotlib jupyter
```

*本仓库仅包含代码和报告，数据文件已通过 .gitignore 排除。*
