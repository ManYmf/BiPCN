import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_experiment_results(
    df: pd.DataFrame,
    title_prefix: str = "Experiment",
    final_metric_key: str = "final/test_acc",
    show_plots: bool = True,
    return_stats: bool = True
) -> dict | None:
    """
    一键生成实验结果的全套可视化图表（适配 epoch 类指标 + 最终指标分布）
    
    参数说明：
    --------
    df: pd.DataFrame
        包含实验结果的DataFrame，需包含：
        - config/seed: 随机种子列
        - epoch_<数字>/<metric>: 各epoch的指标列（如epoch_1/test_acc）
        - final/test_acc: 最终指标列（可通过final_metric_key自定义）
    title_prefix: str
        所有图表标题的前缀（如"MNIST DiscPC"）
    final_metric_key: str
        最终指标的列名，用于找最优seed和绘制分布
    show_plots: bool
        是否显示所有生成的图表（plt.show()）
    return_stats: bool
        是否返回统计结果字典（包含各步骤的统计数据）
    
    返回值：
    -------
    stat_dict: dict (仅当return_stats=True时返回)
        包含所有绘图过程中生成的统计数据，键说明：
        - available_metrics: 可用的epoch指标列表
        - best_seed: 最优种子值
        - test_acc_mean: test_acc的mean±std统计
        - E_train_mean: 训练集能量的mean±std统计
        - E_test_mean: 测试集能量的mean±std统计
        - test_acc_best_over_seeds: 各epoch最优seed的test_acc
        - best_seed_test_acc: 最优seed的test_acc曲线
        - best_seed_energies: 最优seed的能量曲线（train+test）
        - final_metric_vals: 最终指标的分布值
    """
    # -------------------------
    # 内部工具函数
    # -------------------------
    def list_epoch_metrics(df_inner: pd.DataFrame):
        """扫描 df 列名，找出所有 epoch_k/<metric> 的 metric 名称集合"""
        pat = re.compile(r"^epoch_(\d+)/(.*)$")
        metrics = set()
        for c in df_inner.columns:
            m = pat.match(c)
            if m:
                metrics.add(m.group(2))
        return sorted(metrics)

    def _extract_epoch_series(df_inner: pd.DataFrame, metric: str, seed_col="config/seed"):
        """wide -> long，返回 [seed, epoch, value] 的长格式DataFrame"""
        pat = re.compile(rf"^epoch_(\d+)/{re.escape(metric)}$")
        cols, epochs = [], []
        for c in df_inner.columns:
            m = pat.match(c)
            if m:
                cols.append(c)
                epochs.append(int(m.group(1)))

        if not cols:
            raise ValueError(f"No columns for metric='{metric}'. Expected epoch_k/{metric}")

        order = np.argsort(epochs)
        cols = [cols[i] for i in order]
        epochs = [epochs[i] for i in order]

        out = df_inner[[seed_col] + cols].copy()
        out = out.melt(id_vars=[seed_col], var_name="col", value_name="value")
        out["epoch"] = out["col"].str.extract(r"epoch_(\d+)/")[0].astype(int)
        out = out.drop(columns=["col"]).rename(columns={seed_col: "seed"})
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.sort_values(["seed", "epoch"]).reset_index(drop=True)
        return out

    def _mean_std_over_seeds(long_df: pd.DataFrame):
        """按epoch计算mean和std"""
        g = long_df.groupby("epoch")["value"]
        stat = pd.DataFrame({
            "epoch": g.mean().index,
            "mean": g.mean().values,
            "std": g.std(ddof=1).values
        })
        return stat

    def best_seed(df_inner: pd.DataFrame, key=final_metric_key, seed_col="config/seed"):
        """找到最终指标最优的seed"""
        idx = pd.to_numeric(df_inner[key], errors="coerce").idxmax()
        return int(df_inner.loc[idx, seed_col])

    # -------------------------
    # 内部绘图函数
    # -------------------------
    def plot_mean_curve(metric: str, title_suffix: str, ylabel: str, show_std=True):
        """绘制mean±std曲线"""
        long_df = _extract_epoch_series(df, metric)
        stat = _mean_std_over_seeds(long_df)

        plt.figure()
        plt.plot(stat["epoch"], stat["mean"], label="mean")
        if show_std:
            lo = stat["mean"] - stat["std"]
            hi = stat["mean"] + stat["std"]
            plt.fill_between(stat["epoch"], lo, hi, alpha=0.2, label="±1 std")

        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix}: {title_suffix}")
        plt.legend()
        plt.tight_layout()
        return stat

    def plot_best_seed_curve(best_s: int, metric: str, title_suffix: str, ylabel: str):
        """绘制最优seed的指标曲线"""
        sub = df[df["config/seed"] == best_s].copy()
        assert len(sub) == 1, f"Expected exactly 1 row for seed={best_s}, got {len(sub)}"
        long_df = _extract_epoch_series(sub, metric)

        plt.figure()
        plt.plot(long_df["epoch"], long_df["value"])
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix}: {title_suffix} (best seed={best_s})")
        plt.tight_layout()
        return long_df

    def plot_best_over_seeds(metric: str, title_suffix: str, ylabel: str):
        """绘制各epoch所有seed的max值曲线"""
        long_df = _extract_epoch_series(df, metric)
        best = long_df.groupby("epoch")["value"].max().reset_index()

        plt.figure()
        plt.plot(best["epoch"], best["value"], label="best-over-seeds (max)")
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix}: {title_suffix}")
        plt.legend()
        plt.tight_layout()
        return best

    def plot_final_distribution(key: str, title_suffix: str, ylabel: str):
        """绘制最终指标的分布（箱线+散点）"""
        vals = pd.to_numeric(df[key], errors="coerce").dropna().values

        plt.figure()
        plt.boxplot(vals, vert=True)
        plt.scatter(np.ones_like(vals), vals)  # 散点叠加箱线图
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix}: {title_suffix}")
        plt.tight_layout()
        return vals

    # -------------------------
    # 主逻辑：生成所有推荐图表
    # -------------------------
    # 1. 打印可用的epoch指标
    available_metrics = list_epoch_metrics(df)
    print(f"Available epoch metrics: {available_metrics}")

    # 2. 核心指标：test_acc mean±std
    test_acc_mean = plot_mean_curve(
        metric="test_acc",
        title_suffix="test_acc (mean±std over seeds)",
        ylabel="test_acc",
        show_std=True
    )

    # 3. 能量曲线：train/test 能量 mean±std
    E_train_mean = plot_mean_curve(
        metric="E_train_after_infer",
        title_suffix="E_train_after_infer (mean±std)",
        ylabel="E_train",
        show_std=True
    )
    E_test_mean = plot_mean_curve(
        metric="E_test_after_infer",
        title_suffix="E_test_after_infer (mean±std)",
        ylabel="E_test (clamped)",
        show_std=True
    )

    # 4. 各epoch最优seed的test_acc
    test_acc_best_over_seeds = plot_best_over_seeds(
        metric="test_acc",
        title_suffix="test_acc best-over-seeds (max)",
        ylabel="test_acc"
    )

    # 5. 最优seed的详细曲线
    best_s = best_seed(df)
    print(f"Best seed (by {final_metric_key}) = {best_s}")
    
    # 最优seed的test_acc曲线
    best_seed_test_acc = plot_best_seed_curve(
        best_s=best_s,
        metric="test_acc",
        title_suffix="test_acc",
        ylabel="test_acc"
    )

    # 最优seed的能量对比曲线（train+test放同一张图）
    best_E_train = _extract_epoch_series(df[df["config/seed"] == best_s], "E_train_after_infer")
    best_E_test = _extract_epoch_series(df[df["config/seed"] == best_s], "E_test_after_infer")
    
    plt.figure()
    plt.plot(best_E_train["epoch"], best_E_train["value"], label="E_train_after_infer")
    plt.plot(best_E_test["epoch"], best_E_test["value"], label="E_test_after_infer")
    plt.xlabel("epoch")
    plt.ylabel("Energy")
    plt.title(f"{title_prefix}: Energies after inference (best seed={best_s})")
    plt.legend()
    plt.tight_layout()

    # 6. 最终指标的分布
    final_metric_vals = plot_final_distribution(
        key=final_metric_key,
        title_suffix=f"Distribution over seeds: {final_metric_key}",
        ylabel=final_metric_key
    )

    # 显示所有图表
    if show_plots:
        plt.show()

    # 整理返回的统计结果
    if return_stats:
        stat_dict = {
            "available_metrics": available_metrics,
            "best_seed": best_s,
            "test_acc_mean": test_acc_mean,
            "E_train_mean": E_train_mean,
            "E_test_mean": E_test_mean,
            "test_acc_best_over_seeds": test_acc_best_over_seeds,
            "best_seed_test_acc": best_seed_test_acc,
            "best_seed_energies": {
                "E_train": best_E_train,
                "E_test": best_E_test
            },
            "final_metric_vals": final_metric_vals
        }
        return stat_dict

# -------------------------
# 调用示例
# -------------------------
if __name__ == "__main__":
    # 假设你的df已经加载完成（替换成实际的df）
    df = results["df"]  # 你的原始df
    
    # 调用函数生成所有图表
    stats = plot_experiment_results(
        df=df,  # 替换为你的实际DataFrame
        title_prefix="MNIST DiscPC",  # 自定义标题前缀
        final_metric_key="final/test_acc",  # 最终指标列名
        show_plots=True,
        return_stats=True
    )
    
    # 可通过返回的stats查看详细统计数据
    print("最优seed:", stats["best_seed"])
    print("test_acc均值统计:\n", stats["test_acc_mean"].head())