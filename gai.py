"""
增强泛化能力的原型学习 Loss

=== 问题分析 ===
原版原型学习泛化差的原因：
1. 原型过拟合：原型只记住训练集分布，对新分布无适应能力
2. 特征空间不鲁棒：特征对训练数据的噪声和分布特征过度敏感
3. 决策边界过硬：类别之间的边界过于依赖训练集，缺乏弹性

=== 修改方案（5个独立模块，不污染原有代码） ===

【1】特征扰动（Feature Perturbation）
    - 训练时对特征加入高斯噪声，迫使模型学习对微小分布偏移鲁棒的表示
    - 类似 dropout 的思想，但作用在特征值上而非神经元上
    - 测试时关闭，不影响推理

【2】自适应温度（Adaptive Temperature）
    - 固定温度对不同难度的样本一视同仁，容易在简单样本上过拟合
    - 改为可学习温度，让模型自己决定 logits 的锐利程度
    - 加下界防止温度趋近于0导致梯度爆炸

【3】原型正则化（Prototype Spread Regularization）
    - 原版 diversity loss 只防止原型坍缩，但不保证原型均匀分布
    - 新增：惩罚任意两个原型之间余弦相似度超过阈值的情况
    - 效果：原型在超球面上尽可能均匀分布，留出更多空间给未见过的分布

【4】硬负例挖掘（Hard Negative Mining）
    - 原版 compactness loss 对所有样本同等对待
    - 改为重点关注"混淆样本"——那些离错误原型很近的样本
    - 这些样本对泛化最关键：如果模型能区分它们，遇到新分布也更可靠

【5】跨原型对比学习（Cross-Prototype Contrastive Loss）
    - 全新的 loss 分量：对每个样本，拉近正确原型，同时推远所有错误原型
    - 使用 InfoNCE 形式，直接优化特征空间的判别性
    - 与 CE loss 互补：CE 关注分类对错，对比学习关注特征空间结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLossEnhanced(TopVirtualLoss):
    """
    增强泛化的原型学习 Loss

    相比原版新增：
    - feature_noise_std: 特征扰动强度（训练时）
    - learnable_temperature: 是否使用可学习温度
    - spread_threshold: 原型分散正则化的相似度阈值
    - hard_negative_ratio: 硬负例挖掘的比例
    - contrastive_weight: 对比学习 loss 权重

    原有接口不变，init() / forward() / step() 签名兼容
    """

    def __init__(self):
        super().__init__()

    def init(self, input_dim, num_targets, temperature=0.07,
             momentum=0.999, diversity_weight=0.1,
             compactness_weight=0.5, smoothing=0.1,
             # ====== 以下为新增参数 ======
             feature_noise_std=0.1,         # 【1】特征扰动强度，0 表示关闭
             learnable_temperature=True,     # 【2】是否使用可学习温度
             spread_threshold=0.5,           # 【3】原型分散的相似度阈值
             spread_weight=0.1,              # 【3】原型分散正则化权重
             hard_negative_ratio=0.3,        # 【4】硬负例比例
             contrastive_weight=0.3,         # 【5】对比学习权重
             ):

        # ---------- 原型参数 ----------
        self.prototypes = nn.Parameter(torch.empty(num_targets, input_dim))
        nn.init.xavier_uniform_(self.prototypes)

        self.register_buffer('ema_prototypes', self.prototypes.data.clone())
        self.register_buffer('cluster_counts', torch.zeros(num_targets))

        # ---------- 温度 ----------
        # 【2】自适应温度：初始值为传入的 temperature，训练中自动调整
        if learnable_temperature:
            # 用 log 参数化，保证 temperature 始终为正
            self.log_temperature = nn.Parameter(
                torch.tensor(temperature).log()
            )
        else:
            self.register_buffer(
                'log_temperature', torch.tensor(temperature).log()
            )
        self.learnable_temperature = learnable_temperature
        self.min_temperature = 0.01  # 温度下界，防止梯度爆炸

        # ---------- 超参数 ----------
        self.momentum = momentum
        self.diversity_weight = diversity_weight
        self.compactness_weight = compactness_weight
        self.num_targets = num_targets
        self.input_dim = input_dim
        self.loss_function = nn.CrossEntropyLoss(label_smoothing=smoothing)

        # ---------- 新增超参数 ----------
        self.feature_noise_std = feature_noise_std          # 【1】
        self.spread_threshold = spread_threshold            # 【3】
        self.spread_weight = spread_weight                  # 【3】
        self.hard_negative_ratio = hard_negative_ratio      # 【4】
        self.contrastive_weight = contrastive_weight        # 【5】

    @property
    def temperature(self):
        """实际温度值，带下界保护"""
        return torch.clamp(self.log_temperature.exp(), min=self.min_temperature)

    def _perturb_features(self, features):
        """
        【1】特征扰动

        原理：训练时给归一化前的特征加高斯噪声
        - 噪声强度由 feature_noise_std 控制
        - 迫使模型不能依赖特征的精确值，必须学习更鲁棒的模式
        - 类比：图像增强是在输入空间加扰动，这里是在特征空间加扰动
        - 测试时（eval 模式）自动关闭

        为什么有效：
        训练集和测试集的分布差异，在特征空间中体现为特征向量的偏移。
        加噪声模拟了这种偏移，让模型提前"见过"类似的变化。
        """
        if self.training and self.feature_noise_std > 0:
            noise = torch.randn_like(features) * self.feature_noise_std
            features = features + noise
        return features

    def _spread_regularization(self, prototypes_norm):
        """
        【3】原型分散正则化

        原理：不仅防止原型坍缩（diversity loss 已做），还要求原型之间
        的相似度不超过阈值 spread_threshold。

        具体做法：
        - 计算所有原型两两余弦相似度
        - 对超过阈值的相似度对，施加 hinge 惩罚：max(0, sim - threshold)^2
        - 低于阈值的不惩罚（已经足够分散了）

        为什么比原版 diversity loss 更好：
        原版惩罚所有相似度，会浪费优化资源在已经分散的原型上。
        新版只惩罚过近的原型，优化更聚焦，效果更好。

        对泛化的作用：
        原型均匀分布 → 特征空间中类别边界更均匀 → 新分布的数据更容易
        落在正确的原型附近，而不是被挤到两个很近的原型之间产生混淆。
        """
        # (K, K) 相似度矩阵
        proto_sim = torch.mm(prototypes_norm, prototypes_norm.t())
        mask = ~torch.eye(self.num_targets, dtype=torch.bool,
                          device=proto_sim.device)
        similarities = proto_sim[mask]

        # hinge loss：只惩罚超过阈值的
        violations = F.relu(similarities - self.spread_threshold)
        return violations.pow(2).mean()

    def _hard_negative_compactness(self, features_norm, prototypes_norm, targets):
        """
        【4】硬负例挖掘的紧凑性损失

        原理：不是所有样本对泛化都同样重要。
        "硬样本"是那些离正确原型远、或离错误原型近的样本。
        重点优化这些样本，能更有效地利用训练信号。

        具体做法：
        - 计算每个样本到其正确原型的余弦相似度
        - 按相似度从低到高排序（相似度低 = 难样本）
        - 取最难的 hard_negative_ratio 比例的样本
        - 只对这些样本计算 compactness loss

        为什么有效：
        简单样本（已经很靠近正确原型的）贡献的梯度很小且方向噪声大。
        困难样本贡献的梯度大且信息量高。聚焦在困难样本上，
        相当于在训练集的"边界区域"投入更多精力，而这些边界区域
        正是泛化到新分布时最容易出错的地方。
        """
        # 每个样本与其正确原型的相似度
        target_sim = (features_norm * prototypes_norm[targets]).sum(dim=1)

        # 取最难的样本
        num_hard = max(1, int(len(target_sim) * self.hard_negative_ratio))
        hard_indices = target_sim.topk(num_hard, largest=False).indices

        # 只对硬样本计算 compactness
        hard_sim = target_sim[hard_indices]
        return (1.0 - hard_sim).mean()

    def _contrastive_loss(self, features_norm, prototypes_norm, targets):
        """
        【5】跨原型对比学习

        原理：InfoNCE loss 的变体，专门为原型学习设计。

        公式：L = -log( exp(sim(f, p+) / τ) / Σ_k exp(sim(f, p_k) / τ) )

        其中 f 是特征，p+ 是正确原型，p_k 遍历所有原型，τ 是温度。

        与 CE loss 的区别和互补：
        - CE loss 关注"分类是否正确"，只看 logits 的相对大小
        - 对比学习关注"特征空间的结构"，显式拉近正对、推远负对
        - CE loss 可能找到一个"够用"的边界就停了
        - 对比学习会持续优化特征空间，让类内更紧凑、类间更分散

        对泛化的作用：
        更好的特征空间结构 = 类别之间有更大的 margin
        = 新分布的数据即使偏移了，也不容易越过边界被错误分类
        """
        # 所有样本与所有原型的相似度 (B, K)
        all_sim = torch.mm(features_norm, prototypes_norm.t()) / self.temperature

        # 正样本相似度：每个样本与其正确原型
        positive_sim = all_sim[torch.arange(len(targets),
                                            device=targets.device), targets]

        # InfoNCE: -log(exp(pos) / sum(exp(all)))
        # 等价于: -pos + log(sum(exp(all)))
        loss = -positive_sim + torch.logsumexp(all_sim, dim=1)

        return loss.mean()

    def forward(self, inputs, targets=None):
        assert len(inputs.shape) == 3 and inputs.shape[2] == 1
        features = torch.squeeze(inputs, dim=2)  # (B, D)

        # 【1】特征扰动（仅训练时生效）
        features = self._perturb_features(features)

        features_norm = F.normalize(features, dim=1)
        prototypes_norm = F.normalize(self.prototypes, dim=1)

        # 【2】自适应温度
        logits = torch.mm(features_norm, prototypes_norm.t()) / self.temperature

        # 存储 posterior 供外部使用
        self.posterior = logits.unsqueeze(2).detach()

        # 测试阶段只需要 posterior
        if targets is None:
            return None

        # ==================== Loss 计算 ====================

        # (A) 分类损失 — 权重 1.0
        ce_loss = ce_loss = self.loss_function(logits, targets)

        # (B) 原型分散正则化 — 权重 spread_weight
        # 【3】替代原版 diversity loss，只惩罚过近的原型
        spread_loss = self._spread_regularization(prototypes_norm)

        # (C) 硬负例紧凑性损失 — 权重 compactness_weight
        # 【4】替代原版全样本 compactness，聚焦困难样本
        compactness_loss = self._hard_negative_compactness(
            features_norm, prototypes_norm, targets
        )

        # (D) 对比学习损失 — 权重 contrastive_weight
        # 【5】全新分量，优化特征空间结构
        contrastive_loss = self._contrastive_loss(
            features_norm, prototypes_norm, targets
        )

        # ==================== 总 Loss ====================
        total_loss = (ce_loss
                      + self.spread_weight * spread_loss
                      + self.compactness_weight * compactness_loss
                      + self.contrastive_weight * contrastive_loss)

        # 更新统计信息
        with torch.no_grad():
            self.cluster_counts.zero_()
            self.cluster_counts.scatter_add_(
                0, targets, torch.ones_like(targets, dtype=torch.float))

        return total_loss

    @torch.no_grad()
    def step(self, *args, **kwargs):
        """EMA 更新原型"""
        self.ema_prototypes.mul_(self.momentum).add_(
            self.prototypes.data, alpha=1.0 - self.momentum)
        self.prototypes.data.copy_(self.ema_prototypes)


# =============================================================
# 使用示例
# =============================================================
#
# loss_module = PrototypeLossEnhanced()
# loss_module.init(
#     input_dim=128,
#     num_targets=10,
#     temperature=0.07,           # 初始温度（会自动学习调整）
#     feature_noise_std=0.1,      # 特征扰动强度，建议 0.05~0.2
#     learnable_temperature=True, # 开启自适应温度
#     spread_threshold=0.5,       # 原型相似度阈值，建议 0.3~0.6
#     spread_weight=0.1,          # 原型分散权重
#     hard_negative_ratio=0.3,    # 硬负例比例，建议 0.2~0.5
#     contrastive_weight=0.3,     # 对比学习权重，建议 0.1~0.5
#     compactness_weight=0.3,     # 紧凑性权重
#     diversity_weight=0.1,       # 保留但未使用（被 spread 替代）
#     smoothing=0.1,              # label smoothing
# )
#
# # 训练
# loss_module.train()
# loss = loss_module(inputs, targets)
# loss.backward()
# optimizer.step()
# loss_module.step()  # EMA 更新原型
#
# # 测试
# loss_module.eval()
# loss_module(inputs)  # 只生成 posterior，不算 loss
# posterior = loss_module.posterior
#
# =============================================================
# model2enroll.py 的兼容修改（同之前）
# =============================================================
#
# for component in model:
#     if component.split('.')[0] == "loss" or component.split('.')[0] == "loss1":
#         if "ema_prototypes" in component or "cluster_counts" in component:
#             continue
#         if "log_temperature" in component:
#             continue
#         params = model[component].squeeze().cpu().numpy()
