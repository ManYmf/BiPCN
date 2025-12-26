import pandas as pd
import os
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 设置工作目录为当前文件的目录
os.chdir(current_dir)

# 加载数据
df = pd.read_pickle("discpc_baseline_AdamW.pkl")

# 步骤1：定义筛选条件
condition = (df["config/use_bias"] == True) & (df["config/lr_x"] == 1e-2)

# 步骤2：筛选行 + 选取目标列（final/test_acc）
filtered_acc = df[condition]["final/test_acc"]

# 查看筛选后的数据（取消截断，显示所有行）
pd.set_option('display.max_rows', None)
print("筛选后的final/test_acc列数据：")
print(filtered_acc)