import pandas as pd
import os
import numpy as np

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 设置工作目录为当前文件的目录
os.chdir(current_dir)

# 加载数据
df = pd.read_pickle("discpc_baseline_SGD-std.pkl")

# ===================== 关键优化1：明确核心列 =====================
# 确认目标列（你的数据里确实存在）
TARGET_COL = "final/test_acc"
# 手动指定核心超参数列（排除冗余列+处理不可哈希列）
CORE_HYPERPARAMS = [
    "config/batch_size",
    "config/epochs",
    "config/layers",          # 列表类型，后续转字符串
    "config/activation",
    "config/last_send_activation",
    "config/steps_train",
    "config/lr_x",
    "config/momentum_x",
    "config/lr_theta",
    "config/momentum_theta",
    "config/weight_decay"     # 核心优化的超参数
]

# ===================== 关键优化2：数据预处理 =====================
filtered_df = df.copy()

# 1. 检查目标列是否存在且有有效数据
if TARGET_COL not in filtered_df.columns:
    raise ValueError(f"数据中不存在列 {TARGET_COL}！")
if filtered_df[TARGET_COL].isnull().sum() > 0:
    print(f"⚠️ 警告：{TARGET_COL} 列有 {filtered_df[TARGET_COL].isnull().sum()} 个空值，已过滤")
    filtered_df = filtered_df.dropna(subset=[TARGET_COL])
if len(filtered_df) == 0:
    raise ValueError("过滤空值后无有效数据！")

# 2. 处理不可哈希的列（如列表转字符串）
for col in CORE_HYPERPARAMS:
    if filtered_df[col].dtype == 'object' and isinstance(filtered_df[col].iloc[0], list):
        filtered_df[col] = filtered_df[col].astype(str)
        print(f"📌 已将 {col} 列（列表类型）转换为字符串")

# 3. 检查核心超参数列是否都存在
missing_cols = [col for col in CORE_HYPERPARAMS if col not in filtered_df.columns]
if missing_cols:
    raise ValueError(f"数据中缺少核心超参数列：{missing_cols}")

# ===================== 关键优化3：分组统计 =====================
# 按核心超参数分组（大幅减少分组维度）
param_metrics = filtered_df.groupby(CORE_HYPERPARAMS)[TARGET_COL].agg(
    均值='mean',
    最大值='max',
    标准差='std',
    实验次数='count'
).reset_index()

# 调试：显示分组后的基本信息
print(f"\n📊 分组后统计信息：")
print(f"   总分组数：{len(param_metrics)}")
print(f"   各分组实验次数分布：")
print(param_metrics["实验次数"].value_counts().sort_index())

# ===================== 关键优化4：鲁棒性处理 =====================
if param_metrics.empty:
    print("❌ 分组后无数据！")
else:
    # 按均值降序排序
    best_param_df = param_metrics.sort_values(by="均值", ascending=False)
    
    # 输出所有分组（显示前20行，避免刷屏）
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f"{x:.6f}")
    
    print("\n=== 核心超参数组合的test_acc表现（按均值降序，显示前20行）===")
    print(best_param_df)
    
    # 提取最佳组合
    best_params = best_param_df.iloc[0].copy()
    best_acc_mean = best_params["均值"]
    best_acc_max = best_params["最大值"]
    best_acc_std = best_params["标准差"] if not np.isnan(best_params["标准差"]) else 0.0
    best_exp_count = best_params["实验次数"]
    
    # 输出最佳组合详情
    print("\n=== 最佳超参数组合结果 ===")
    print("📌 核心超参数组合：")
    for col in CORE_HYPERPARAMS:
        print(f"   {col}: {best_params[col]}")
    
    print("\n📈 性能指标（基于所有seed）：")
    print(f"   平均test_acc: {best_acc_mean:.6f} ({best_acc_mean*100:.2f}%)")
    print(f"   最高test_acc: {best_acc_max:.6f} ({best_acc_max*100:.2f}%)")
    print(f"   标准差: {best_acc_std:.6f}")
    print(f"   实验次数: {best_exp_count}")
    
    # 输出该组合下的所有seed数据
    print("\n📝 最佳组合下各seed的原始test_acc：")
    best_filter = True
    for col in CORE_HYPERPARAMS:
        best_filter = best_filter & (filtered_df[col] == best_params[col])
    best_seed_df = filtered_df[best_filter][["config/seed", TARGET_COL]].sort_values(by="config/seed")
    print(best_seed_df)
    
    # LaTeX格式输出
    print("\n=== LaTeX格式结果 ===")
    acc_mean_pct = best_acc_mean * 100
    acc_std_pct = best_acc_std * 100
    latex_str = f"{acc_mean_pct:.2f} \\pm {acc_std_pct:.2f}\\%"
    print(f"   均值±标准差：{latex_str}")