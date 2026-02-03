═══════════════════════════════════════════════════════════════════════════════
                          深度学习推荐模型完整架构
═══════════════════════════════════════════════════════════════════════════════

输入层
─────────────────────────────────────────────────────────────────────────────
特征字段 (Field)：33个字段
  ├─ f0_508, f0_301, f0_205, f0_206, f0_210, f0_509, f0_216, f0_207, f0_702, f0_853
  └─ f1_121, f1_129, f1_110_14, f1_127, f1_127_14, f1_124, f1_150_14, f1_109_14, f1_126, f1_101, f1_125, f1_128, f1_122, ...


Embedding 层 (FeatureEmbedding)
─────────────────────────────────────────────────────────────────────────────
每个特征字段有独立的 Embedding 表：
  
  特征            | Embedding维度 | Embedding表大小 | 模式      | 输出形状
  ────────────────┼──────────────┼────────────────┼──────────┼──────────
  f0_508          |     8        | 81,204+1       | mean     | [B, 8]
  f0_301          |     8        | 1,931          | mean     | [B, 8]
  f0_205          |     8        | 3,372,175,220  | mean     | [B, 8]
  f0_206          |     8        | 81,204+1       | mean     | [B, 8]
  f0_210          |     8        | 43,493,988+1   | mean     | [B, 8]
  f0_509          |     8        | 1,685,967,244+1| mean     | [B, 8]
  f0_216          |     8        | ...            | mean     | [B, 8]
  f0_207          |     8        | ...            | mean     | [B, 8]
  f0_702          |     8        | ...            | mean     | [B, 8]
  f0_853          |     8        | ...            | mean     | [B, 8]  (multi-hot)
  
  f1_121          |     8        | 36,042+1       | mean     | [B, 8]
  f1_129          |     8        | 2,412+1        | mean     | [B, 8]
  f1_110_14       |     8        | ...            | mean     | [B, 8]  (multi-hot, weighted)
  f1_127          |     8        | 2,010+1        | mean     | [B, 8]
  f1_127_14       |     8        | 1,686,040,117+1| mean     | [B, 8]  (multi-hot, weighted)
  f1_124          |     8        | 1,608+1        | mean     | [B, 8]
  f1_150_14       |     8        | ...            | mean     | [B, 8]  (multi-hot, weighted)
  f1_109_14       |     8        | ...            | mean     | [B, 8]  (multi-hot, weighted)
  f1_126          |     8        | ...            | mean     | [B, 8]
  f1_101          |     8        | ...            | mean     | [B, 8]
  f1_125          |     8        | ...            | mean     | [B, 8]
  f1_128          |     8        | ...            | mean     | [B, 8]
  f1_122          |     8        | ...            | mean     | [B, 8]
  ... (约33个字段共计)

输出：
  emb_stack:      [B, 33, 8]       # embedding 栈
  emb_concat:     [B, 33×8=264]    # embedding 拼接
  sum_emb:        [B, 8]           # embedding 求和
  mean_emb:       [B, 8]           # embedding 平均


Linear (一阶) 层
─────────────────────────────────────────────────────────────────────────────
为每个特征字段创建 EmbeddingBag(embedding_dim=1)
  ├─ 共 33 个 linear embeddings，每个输出 [B, 1]
  └─ 输出: logit_linear [B, 1]  (所有 linear 项求和)


FM (二阶) 层
─────────────────────────────────────────────────────────────────────────────
输入: emb_stack [B, 33, 8]

计算过程:
  sum_emb = sum(emb_stack, dim=1)              # [B, 8]
  sum_square = sum_emb * sum_emb               # [B, 8]
  square_sum = sum(emb_stack^2, dim=1)         # [B, 8]
  
  fm_vec = 0.5 * (sum_square - square_sum)     # [B, 8]
  fm_vec = fm_vec / 33                         # 按字段数归一化

输出: fm_vec [B, 8]


Deep 层 (MLP)
─────────────────────────────────────────────────────────────────────────────
输入: emb_concat [B, 264]

网络结构:
  Linear(264 → 256)
    ├─ ReLU激活
    ├─ Dropout(0.2)
  Linear(256 → 128)
    ├─ ReLU激活
    ├─ Dropout(0.2)

输出: deep_h [B, 128]
参数量: 264×256 + 256 + 256×128 + 128 = 99,712 参数


Output Projection (Backbone)
─────────────────────────────────────────────────────────────────────────────
Legacy DeepFM Mode (use_legacy_pseudo_deepfm=True)：
  
  输入: concat(deep_h[128], fm_vec[8]) = [B, 136]
  Linear(136 → 128)
  
  输出: h [B, 128]


MMoE 输入组装 (MMoEInputComposer)
─────────────────────────────────────────────────────────────────────────────
根据配置:
  add_fm_vec:      True
  add_emb:         "mean"
  part_proj_dim:   128
  fusion:          "concat"
  adapter_mlp_dims: [128]
  norm:            "layernorm"

组装的部分:
  ├─ deep_h [B, 128]           (Part 1: Deep representation)
  │   ├─ Linear(128 → 128)
  │   ├─ LayerNorm(128)
  │   └─ Dropout(0.2)
  │   输出: [B, 128]
  │
  ├─ fm_vec [B, 8]             (Part 2: FM features)
  │   ├─ Linear(8 → 128)      (投影)
  │   ├─ LayerNorm(128)
  │   └─ Dropout(0.2)
  │   输出: [B, 128]
  │
  └─ mean_emb [B, 8]           (Part 3: Embedding aggregation)
      ├─ Linear(8 → 128)      (投影)
      ├─ LayerNorm(128)
      └─ Dropout(0.2)
      输出: [B, 128]

融合 (concat):
  concat([part1, part2, part3])  # [B, 128+128+128] = [B, 384]

Adapter MLP:
  Linear(384 → 128)
    ├─ ReLU激活
    └─ Dropout(0.1)
  Linear(128 → 128)
  
  输出: [B, 128]

MMoE输入最终维度: 128


MMoE 主模块 (Mixture of Experts)
─────────────────────────────────────────────────────────────────────────────
输入: x [B, 128]

LayerNorm: x = LayerNorm(x)  # [B, 128]


专家网络 (4个独立MLP)
────────────────────
每个专家:
  Expert[i]:
    Linear(128 → 128)
      ├─ ReLU激活
      └─ Dropout(0.2)
    输出: [B, 128]

4个专家的输出: [Expert1, Expert2, Expert3, Expert4]，都是 [B, 128]
expert_outputs_stack: [B, 128, 4]


门控网络 (Per-Task Gating)
────────────────────────────
两个任务（CTR、CVR）各一个门：

Gate_CTR:
  输入: x [B, 128]
  Linear(128 → 4)  
  + Temperature scaling (T=1.2)
  + 训练时高斯噪声 (std=0.05)
  Softmax(dim=-1)
  
  输出: gate_weights_ctr [B, 4]
  
Gate_CVR:
  输入: x [B, 128]
  Linear(128 → 4)
  + Temperature scaling (T=1.2)
  + 训练时高斯噪声 (std=0.05)
  Softmax(dim=-1)
  
  输出: gate_weights_cvr [B, 4]

注：门控稳定化配置:
  ├─ Temperature: 1.2 (使路由分布更平滑)
  ├─ Noise std: 0.05 (避免路由锁死)
  ├─ 熵正则: 1.0e-3 (鼓励更均匀的路由)
  └─ 负载均衡KL: 1.0e-3 (促进各专家负载均衡)


混合步骤
────────
对于 CTR 任务:
  stacked = stack(expert_outputs, dim=2)    # [B, 128, 4]
  weights = gate_weights_ctr.unsqueeze(1)   # [B, 1, 4]
  mixed_ctr = bmm(stacked, weights^T)        # [B, 128, 1]
  mixed_ctr = squeeze(-1)                    # [B, 128]

对于 CVR 任务:
  mixed_cvr = bmm(stacked, gate_weights_cvr^T)  # [B, 128]


任务塔 (Task Heads)
─────────────────────────────────────────────────────────────────────────────

CTR 头:
  输入: mixed_ctr [B, 128]
  MLP:
    Linear(128 → 128)
      ├─ ReLU激活
      └─ Dropout(0.1)
    Linear(128 → 1)
  
  输出: logit_ctr [B]
  
  参数: 128×128 + 128 + 128 + 1 = 16,513 参数
  
  偏置初始化: bias = log(pos_rate / (1-pos_rate))
           = log(0.07 / 0.93) ≈ -2.57

CVR 头:
  输入: mixed_cvr [B, 128]
  MLP:
    Linear(128 → 128)
      ├─ ReLU激活
      └─ Dropout(0.1)
    Linear(128 → 1)
  
  输出: logit_cvr [B]
  
  参数: 128×128 + 128 + 128 + 1 = 16,513 参数
  
  偏置初始化: bias = log(pos_rate / (1-pos_rate))
           = log(0.01 / 0.99) ≈ -4.6


输出处理
─────────────────────────────────────────────────────────────────────────────
logit_ctr:      [B]  (原始logits)
logit_cvr:      [B]  (原始logits)
logit_linear:   [B, 1] → [B]  (一阶特征的贡献)


════════════════════════════════════════════════════════════════════════════════
                              损失函数计算
════════════════════════════════════════════════════════════════════════════════

ESMM 模式 (use_esmm=True)
────────────────────────────

p_ctr = sigmoid(logit_ctr)              # CTR预测概率 ∈ (0, 1)
p_cvr = sigmoid(logit_cvr)              # CVR预测概率 ∈ (0, 1)

CTCVR 概率（log域计算保证数值稳定）:
  log_p_ctr = logsigmoid(logit_ctr)
  log_p_cvr = logsigmoid(logit_cvr)
  log_p_ctcvr = log_p_ctr + log_p_cvr  # log(p_ctr * p_cvr)
  
  p_ctcvr = exp(log_p_ctcvr) = p_ctr * p_cvr


损失项组合:
───────────

1️⃣  CTR 损失 (BCE with pos_weight):
   ────────────────────────────────
   pos_weight_ctr = 24.7  (静态)
   L_ctr = BCEWithLogits(logit_ctr, y_ctr, pos_weight=24.7)
         = mean(-y_ctr * log(p_ctr) - (1-y_ctr) * log(1-p_ctr)) × 权重
   
   λ_ctr = 1.0


2️⃣  CTCVR 损失 (BCE + 可选Aux-Focal):
   ────────────────────────────────────
   pos_weight_ctcvr = 4800  (静态)
   
   ▪ BCE项:
     L_ctcvr_bce = 
       weighted_BCE(log_p_ctcvr, y_ctcvr, pos_weight=4800)
     
     其中 BCE 从 log-probability 计算:
       BCE = -y * log(p) - (1-y) * log(1-p)
           = -y * log_p_ctcvr - (1-y) * log1mexp(log_p_ctcvr)
   
   ▪ Aux-Focal项 (warmup_steps=8000):
     当 global_step >= 8000 时启用:
       focal_weight = (1 - p_t)^γ,  γ = 1.0
       p_t = p*y + (1-p)*(1-y)  (预测类别的概率)
       
       L_ctcvr_focal = focal_weight × L_ctcvr_bce
       
       注: detach_p_for_weight=True 时，focal_weight不参与梯度
   
   ▪ 合并:
     L_ctcvr = L_ctcvr_bce + λ_focal × L_ctcvr_focal
     其中 λ_focal = 0.05
   
   λ_ctcvr = 1.0


3️⃣  CVR 辅助损失 (click=1 子集):
   ─────────────────────────────
   (仅当有点击的样本时计算)
   
   mask = click_mask  (表示是否曝光且有点击)
   mask_sum = count(mask > 0)
   
   当 mask_sum > 0:
     L_cvr_aux = 
       mean(BCEWithLogits(logit_cvr, y_cvr, reduction='none') * mask)
     
   λ_cvr_aux = 0.1


总损失:
──────
L_total = λ_ctr × L_ctr 
        + λ_ctcvr × L_ctcvr 
        + λ_cvr_aux × L_cvr_aux
        
        = 1.0 × L_ctr + 1.0 × L_ctcvr + 0.1 × L_cvr_aux

    + Gate_Regularization (仅训练时):
        L_gate_reg = entropy_weight × H(gate_w) 
                   + lb_kl_weight × KL(mean(gate_w) || Uniform)
        
        其中:
          H(w) = -Σ(w * log(w + eps))  (熵，鼓励平滑)
          KL(...) (负载均衡，避免某些专家未被利用)


AMP (Automatic Mixed Precision):
──────────────────────────────────
├─ CTR/CTCVR/CVR 损失计算使用 FP32 (数值稳定)
├─ Focal weight 计算使用 FP32 (detach_p_for_weight=True)
└─ 其他计算使用 AMP (自动选择精度)


════════════════════════════════════════════════════════════════════════════════
                            主要 Metrics 计算
════════════════════════════════════════════════════════════════════════════════

Validation 时:
──────────────
• auc_ctr:     计算 CTR 预测 vs 真实标签的 AUC
• auc_cvr:     计算 CVR 预测 vs 真实标签的 AUC（仅点击=1子集）
• auc_ctcvr:   计算 CTCVR 预测 vs 真实标签的 AUC

• auc_primary: 
  ├─ ESMM 模式:
  │   auc_primary = 1.0 × auc_ctcvr + 0.2 × auc_ctr + 0.2 × auc_cvr
  │   
  └─ 非ESMM 模式:
      auc_primary = (auc_ctr + auc_cvr) / 2


════════════════════════════════════════════════════════════════════════════════
                           参数统计汇总
════════════════════════════════════════════════════════════════════════════════

组件                          参数量
────────────────────────────────────
Embedding 层:                 ~30-50M (由 embedding 表大小决定)
  - 线性部分 EmbeddingBag:    33 个表
  - 特征部分 Embedding:       33 个表 × 8维 = 264维拼接

Deep MLP (264→128):           99,712
  Linear(264→256):  67,840
  Linear(256→128):  32,896

Backbone Output (136→128):    17,408
  Linear(136→128):  17,408

FM 计算层:                    0 (仅前向计算，无参数)

MMoE 输入组装:                ~50K
  三个 PartProjector:
    Linear(128→128): 3 × 16,512 = 49,536
    LayerNorm:       3 × 256 = 768
  Adapter MLP:
    Linear(384→128): 49,152
    Linear(128→128): 16,512

MMoE Experts (4个):          ~133K
  4 × Linear(128→128): 4 × 16,512 = 66,048
  LayerNorm:           16,384

MMoE Gates (CTR + CVR):      ~1K
  Gate_CTR Linear(128→4):    516
  Gate_CVR Linear(128→4):    516

CTR 头:                       16,513
  Linear(128→128): 16,512
  Linear(128→1):   129

CVR 头:                       16,513
  Linear(128→128): 16,512
  Linear(128→1):   129

────────────────────────────────────
总计（不含Embedding表）:       ~350K 参数
总计（含Embedding表）:         ~30-50M 参数


════════════════════════════════════════════════════════════════════════════════
                         优化器与学习率调度
════════════════════════════════════════════════════════════════════════════════

Dense Optimizer (DenseAdamW):
  lr:          0.0006
  weight_decay: 1e-5
  betas:       (0.9, 0.999)
  eps:         1e-8

Sparse Optimizer:
  lr:          0.001
  betas:       (0.9, 0.999)
  eps:         1e-8

学习率调度 (Cosine Annealing):
  warmup_steps:   5,000
  total_steps:    40,000
  min_lr_ratio:   0.1
  
  lr(t) = lr_base × [0.5 × (1 + cos(π × (t-warmup) / (total-warmup)))]
        × 在 warmup 期间从 0 线性增长到 lr_base

pos_weight 调度 (Piecewise):
  target:     "ctcvr"
  milestones: [0, 8000]
  values:     [300, 30]
  
  含义:
    0 ≤ step < 8000:  clip(pos_weight, max=300)
    step ≥ 8000:      clip(pos_weight, max=30)


experiment:
  name: "deepfm_mmoe_dual_sparse"

# ESMM toggle (default off to preserve legacy behaviour)
use_esmm: true
esmm:
  version: "v2"  # "v2" (standard ESMM: p_ctcvr = p_ctr * p_cvr) or "legacy" (deprecated)
  eps: 1.0e-8
  lambda_ctr: 1.0
  lambda_ctcvr: 1.0
  lambda_cvr_aux: 0.1  # CVR auxiliary loss on click=1 subset; 0 to disable

data:
  metadata_path: "data/processed/metadata.json"
  batch_size: 1024
  num_workers: 6
  num_workers_valid: 2  # Use fewer workers for validation
  prefetch_factor: 4
  pin_memory: true
  persistent_workers: true
  drop_last: false
  debug: false
  seed: 20260127
  neg_keep_prob_train: 0.4

sampling:
  negative_sampling: "none"  # set to "none" when use_esmm=true

embedding:
  default_embedding_dim: 8
  embedding_dim_overrides: {}
  mode: "sum"  # 其实是mean，只是因为iterdataset不允许，后面转为手动实现了
  sparse_grad: true

model:
  name: "deepfm_mmoe"
  mtl: "mmoe"
  enabled_heads: ["ctr", "cvr"]
  tasks: ["ctr", "cvr"]
  backbone:
    use_legacy_pseudo_deepfm: false
    return_logit_parts: true
    per_head_add:
      ctr: { use_wide: true, use_fm: true }
      cvr: { use_wide: false, use_fm: false }
    deep_hidden_dims: [128, 64]
    deep_dropout: 0.2
    deep_activation: relu
    deep_use_bn: false
    fm_enabled: true
    fm_projection_dim: 16
    out_dim: 64
  mmoe:
    num_experts: 4
    expert_mlp_dims: [128]
    gate_type: "linear"
    gate_hidden_dims: []
    input: "deep_h"
    dropout: 0.2
    log_gates: true  # Enable gate weight logging for health metrics
    gate_stabilize:
      enabled: true
      temperature: 1.2          # >1 更平滑，避免路由锁死
      noise_std: 0.05           # gate logits 加高斯噪声（仅训练时）
      eps: 1.0e-8
      entropy_reg_weight: 1.0e-3      # 熵正则：鼓励更均匀的路由
      load_balance_kl_weight: 1.0e-3  # 负载均衡KL正则
      log_stats: true
    # ===== 新增字段 =====
    add_fm_vec: true       # 是否加入 fm_vec（FM 二阶向量）
    add_emb: "mean"         # embedding 聚合方式: none|sum|mean|concat
    part_proj_dim: 128      # 每个 part 投影到的统一维度；null 表示保留原维度
    fusion: "concat"        # 融合方式: concat|sum
    adapter_mlp_dims: [128]    # 融合后可选 MLP 层维度
    norm: "layernorm"            # 各 part 投影后的归一化: none|layernorm
  heads:
    default:
      mlp_dims: [128]
      dropout: 0.1
      use_bn: false
      activation: "relu"
    ctr: {}
    cvr: {}
  deep_hidden_dims: [256, 128]
  deep_dropout: 0.2
  deep_activation: "relu"
  deep_use_bn: false
  fm_enabled: true
  fm_projection_dim: 16
  out_dim: 128

optim:
  type: dual_sparse_dense
  dense:
    lr: 0.0006
    weight_decay: 1e-5
    betas: [0.9, 0.999]
    eps: 1.0e-8
  sparse:
    enabled: true
    lr: 0.001
    betas: [0.9, 0.999]
    eps: 1.0e-8
    allow_fallback_if_empty: false
  # ============================================================
  # 改动 A: 学习率调度配置
  # ============================================================
  lr_scheduler:
    enabled: true
    target: "both"         # "dense" 只调度 dense optimizer | "both" 同时调度
    type: "cosine"          # "cosine" 余弦退火 | "step" 阶梯衰减
    warmup_steps: 5000       # warmup 步数（smoke test 用小值）
    total_steps:  40000  # 总步数（可不填，自动推导）
    min_lr_ratio: 0.1       # cosine 最小 lr = base_lr * min_lr_ratio
    # step 方案参数（当 type="step" 时生效）
    # step_milestones: [2000, 4000]
    # step_gamma: 0.3

loss:
  w_ctr: 1.0
  w_cvr: 1.0
  eps: 1.0e-6
  pos_weight_dynamic: false
  static_pos_weight:
    ctr: 24.7          #24.7
    ctcvr: 4800           #4800
  pos_weight_clip:
    ctr: 30
    ctcvr: 300
  # ============================================================
  # 改动 B: pos_weight_clip 调度配置
  # ============================================================
  pos_weight_clip_schedule:
    enabled: true
    target: "ctcvr"          # "ctcvr" | "ctr" | "both"
    type: "piecewise"        # "piecewise" 分段 | "linear" 线性
    # piecewise 方案：按里程碑切换
    milestones: [0, 8000]     # step 边界（smoke test 用小值）
    values: [300, 30]       # 对应的 clip 值
    # linear 方案参数（当 type="linear" 时生效）
    # start_step: 0
    # end_step: 2000
    # start_value: 400
    # end_value: 100

  # ===== 方案1: ESMM 主链路 BCE + CTCVR Aux-Focal（配置化 + warmup）=====
  # 说明：
  #   - CTR 损失保持 BCEWithLogitsLoss(pos_weight=24.7)
  #   - CTCVR 损失 = BCE(pos_weight=4800) + aux_focal
  #   - aux_focal 只作用于 CTCVR（不影响 CTR）
  #   - warmup_steps 前不启用 focal，避免初期梯度不稳
  #   - 关闭时 (enabled=false)，行为与原实现完全一致
  # Sweep 建议：
  #   - lambda: 0.05, 0.1, 0.2
  #   - gamma: 1.0, 2.0
  #   - warmup_steps: 1000, 2000
  aux_focal:
    enabled: true              # 开关：true 启用，false 禁用（默认禁用以保持向后兼容）
    warmup_steps: 8000         # warmup 步数：前 N step 不启用 aux focal
    target_head: "ctcvr"       # 目标 head（固定为 ctcvr，不要改为 ctr）
    lambda: 0.05                # focal 辅助项系数（推荐 sweep: 0.05/0.1/0.2）
    gamma: 1.0                 # focal gamma（推荐 sweep: 1.0/2.0；gamma 越大，easy samples 权重越低）
    use_alpha: false           # 是否使用 alpha 平衡（避免与 pos_weight 叠加）
    alpha: 0.25                # alpha 值（仅 use_alpha=true 时生效）
    detach_p_for_weight: true  # focal 权重计算时 detach 概率梯度（推荐 true）
    compute_fp32: true         # AMP 下权重计算使用 fp32（推荐 true，更稳定）
    log_components: true       # 日志中记录 loss_ctcvr_bce 和 loss_ctcvr_focal

runtime:
  device: "cuda"
  epochs: 1
  log_every: 5000
  amp: true
  grad_diag_enabled: true  # Enable new dynamic gradient diagnostics
  grad_diag_every: 1000    # Run at same frequency as logging (null = use log_every)
  grad_diag_min_tasks: 2   # Minimum tasks for shared param identification
  log_health_metrics: true  # Enable health metrics logging
  max_train_steps: 40000
  max_valid_steps: null
  grad_clip_norm: 2.0
  seed: 20260127
  resume_path: null
  auto_eval: true
  auto_eval_split: "valid"
  auto_eval_save_preds: true
  auto_eval_ckpt: "best"
  save_last: false
