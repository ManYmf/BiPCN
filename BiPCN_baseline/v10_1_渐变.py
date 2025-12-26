import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 0) 输入：df
# -----------------------------
# 方式A：你在 notebook / 脚本里已经有 results["df"]
# df = results["df"]

# 方式B：你提前把 df 保存成 CSV，这里直接读
# df = pd.read_csv("results_df.csv")

def _find_alpha_col(df: pd.DataFrame) -> str:
    if "config/alpha_gen" in df.columns:
        return "config/alpha_gen"
    # 兜底：找包含 alpha_gen 的 config 列
    cand = [c for c in df.columns if c.startswith("config/") and "alpha_gen" in c]
    if not cand:
        raise ValueError("Cannot find alpha_gen column. Expected 'config/alpha_gen' in df.columns.")
    return cand[0]

def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _extract_epoch_series(df: pd.DataFrame, metric: str, id_cols):
    """
    将 wide 的 epoch_k/metric 列抽成 long:
    return: long_df with columns id_cols + ["epoch","value"]
    """
    pat = re.compile(rf"^epoch_(\d+)/{re.escape(metric)}$")
    cols = []
    epochs = []
    for c in df.columns:
        m = pat.match(c)
        if m:
            cols.append(c)
            epochs.append(int(m.group(1)))
    if not cols:
        raise ValueError(f"No columns found for metric='{metric}'. Expected like epoch_k/{metric}")

    order = np.argsort(epochs)
    cols = [cols[i] for i in order]
    epochs = [epochs[i] for i in order]

    out = df[id_cols + cols].copy()
    out = out.melt(id_vars=id_cols, var_name="col", value_name="value")
    out["epoch"] = out["col"].str.extract(r"epoch_(\d+)/")[0].astype(int)
    out = out.drop(columns=["col"])
    out["value"] = _to_float_series(out["value"])
    out = out.dropna(subset=["value"])
    return out.sort_values(id_cols + ["epoch"]).reset_index(drop=True)

def _agg_over_seeds(df_sub: pd.DataFrame, value_col: str):
    """
    对同一个 alpha_gen 下的多条 trial（通常不同 seed）聚合。
    """
    g = df_sub.groupby("alpha_gen")[value_col]
    out = pd.DataFrame({
        "mean": g.mean(),
        "std": g.std(ddof=1),
        "n": g.size(),
        "min": g.min(),
        "max": g.max(),
    }).reset_index()
    # 若 n=1，std 会是 NaN；画图时处理掉
    return out

def _errorbar_or_plot(x, y, yerr, label=None):
    yerr_use = None if (yerr.isna().all() if isinstance(yerr, pd.Series) else np.all(np.isnan(yerr))) else yerr
    if yerr_use is None:
        plt.plot(x, y, marker="o", label=label)
    else:
        plt.errorbar(x, y, yerr=yerr_use, marker="o", capsize=3, label=label)

# -----------------------------
# 1) 清洗 alpha_gen / seed
# -----------------------------
alpha_col = _find_alpha_col(df)
df = df.copy()
df["alpha_gen"] = _to_float_series(df[alpha_col])

seed_col = "config/seed" if "config/seed" in df.columns else None
if seed_col is not None:
    df["seed"] = pd.to_numeric(df[seed_col], errors="coerce")
else:
    df["seed"] = np.nan  # 没有 seed 也能跑，只是 std 没意义

df = df.dropna(subset=["alpha_gen"])
df = df.sort_values(["alpha_gen", "seed"]).reset_index(drop=True)

# -----------------------------
# 2) 画：final/test_acc vs alpha_gen
# -----------------------------
if "final/test_acc" not in df.columns:
    raise ValueError("df has no column 'final/test_acc'")

df["final_test_acc"] = _to_float_series(df["final/test_acc"])
acc_stat = _agg_over_seeds(df.dropna(subset=["final_test_acc"]), "final_test_acc")

plt.figure()
_errorbar_or_plot(
    acc_stat["alpha_gen"], acc_stat["mean"], acc_stat["std"],
    label="final/test_acc (mean±std over seeds)"
)
plt.xlabel("alpha_gen")
plt.ylabel("final/test_acc")
plt.title("Alpha sweep: final test accuracy vs alpha_gen")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# -----------------------------
# 3) 画：final/E_test_after_infer vs alpha_gen（如果有）
# -----------------------------
if "final/E_test_after_infer" in df.columns:
    df["final_E_test"] = _to_float_series(df["final/E_test_after_infer"])
    E_stat = _agg_over_seeds(df.dropna(subset=["final_E_test"]), "final_E_test")

    plt.figure()
    _errorbar_or_plot(
        E_stat["alpha_gen"], E_stat["mean"], E_stat["std"],
        label="final/E_test_after_infer (mean±std)"
    )
    plt.xlabel("alpha_gen")
    plt.ylabel("final/E_test_after_infer")
    plt.title("Alpha sweep: final test energy vs alpha_gen")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

# -----------------------------
# 4) 画：test_acc 随 epoch 的曲线（每个 alpha_gen 一条曲线）
# -----------------------------
# 如果你想看“alpha_gen 从 0.1 到 1.0 的整个训练过程”，这一张最直观
id_cols = ["alpha_gen"]
if seed_col is not None:
    id_cols.append("seed")

test_acc_long = _extract_epoch_series(df, metric="test_acc", id_cols=id_cols)

# 对每个 alpha_gen、每个 epoch 聚合 seed
g = test_acc_long.groupby(["alpha_gen", "epoch"])["value"]
curve = pd.DataFrame({
    "mean": g.mean(),
    "std": g.std(ddof=1),
    "n": g.size(),
}).reset_index()

plt.figure()
for a in sorted(curve["alpha_gen"].unique()):
    sub = curve[curve["alpha_gen"] == a].sort_values("epoch")
    plt.plot(sub["epoch"], sub["mean"], label=f"alpha_gen={a:g}")
    # 可选阴影（仅当有多个 seed）
    if (sub["n"].max() >= 2) and (not sub["std"].isna().all()):
        lo = sub["mean"] - sub["std"]
        hi = sub["mean"] + sub["std"]
        plt.fill_between(sub["epoch"], lo, hi, alpha=0.12)

plt.xlabel("epoch")
plt.ylabel("test_acc (mean over seeds)")
plt.title("Alpha sweep: test_acc trajectories over epochs")
plt.grid(True, alpha=0.3)
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()

# -----------------------------
# 5) 导出汇总表（方便你写报告/选最优）
# -----------------------------
summary = acc_stat.rename(columns={"mean": "final_test_acc_mean", "std": "final_test_acc_std",
                                   "min": "final_test_acc_min", "max": "final_test_acc_max", "n": "n_trials"})
if "final/E_test_after_infer" in df.columns:
    summary = summary.merge(
        E_stat.rename(columns={"mean":"final_E_test_mean","std":"final_E_test_std","min":"final_E_test_min","max":"final_E_test_max","n":"n_trials_E"}),
        on="alpha_gen", how="left"
    )
summary.to_csv("alpha_sweep_summary.csv", index=False)
print("Saved:", "alpha_sweep_summary.csv")

plt.show()
