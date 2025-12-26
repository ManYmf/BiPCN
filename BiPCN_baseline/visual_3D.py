import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_alpha_2d_contours(
    df: pd.DataFrame,
    x_param: str = "alpha_gen",
    y_param: str = "alpha_disc",
    z_metric: str = "final/test_acc",
    seed_col: str = "config/seed",
    agg: str = "mean",                # "mean" or "median"
    levels: int = 20,
    use_log_x: bool = False,
    use_log_y: bool = False,
    show_points: bool = True,
    annotate_best: bool = True,
    best_mode: str = "max",           # "max" for acc, "min" for RMSE
    make_std_plot: bool = True,
    save_summary_csv: str | None = None,
    save_fig_prefix: str | None = None,
):
    """
    画二维等高线：z_metric over (config/x_param, config/y_param).

    Parameters
    ----------
    df : pd.DataFrame
        results["df"] 的表（每行=一个 trial，如不同 seed / 不同 alpha 组合）。
    x_param, y_param : str
        config 字段名（不带 "config/" 前缀也可以）。默认 alpha_gen / alpha_disc。
    z_metric : str
        要画的指标列名，默认 "final/test_acc"。也可用 "final/gen_rmse_classavg" 等。
    seed_col : str
        seed 列名（用于 std/n）。若没有也能画，只是 std 可能为 NaN。
    agg : str
        组合内聚合方式："mean" 或 "median"。
    levels : int
        等高线级数。
    use_log_x, use_log_y : bool
        对 x/y 轴做 log10 变换（适合扫数量级，例如 1e-4 到 1）。
    show_points : bool
        在等高线图上叠加采样点散点。
    annotate_best : bool
        标注最优点（默认按 best_mode）。
    best_mode : str
        "max" 选最大值（准确率）；"min" 选最小值（RMSE/损失类指标）。
    make_std_plot : bool
        如果同一 (x,y) 下有多个 seed，会额外画 std 等高线。
    save_summary_csv : str | None
        若给路径，则保存聚合汇总表 CSV。
    save_fig_prefix : str | None
        若给前缀，则保存图像为 f"{prefix}_z.png" / f"{prefix}_std.png"

    Returns
    -------
    summary : pd.DataFrame
        聚合后的长表：含 x,y,z,std,n,min,max
    """

    def _require_col(df_, name):
        if name not in df_.columns:
            raise ValueError(f"Missing column '{name}'. Available columns: {list(df_.columns)[:40]} ...")

    def _to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def _find_config_col(df_, param_name):
        # allow passing "config/alpha_gen" or "alpha_gen"
        if param_name.startswith("config/"):
            key = param_name
        else:
            key = f"config/{param_name}"
        if key in df_.columns:
            return key
        # fallback: fuzzy search
        cand = [c for c in df_.columns if c.startswith("config/") and param_name in c]
        if not cand:
            raise ValueError(f"Cannot find config column for '{param_name}'. Expected '{key}'.")
        return cand[0]

    x_col = _find_config_col(df, x_param)
    y_col = _find_config_col(df, y_param)
    _require_col(df, z_metric)

    d = df.copy()
    d[x_col] = _to_num(d[x_col])
    d[y_col] = _to_num(d[y_col])
    d[z_metric] = _to_num(d[z_metric])
    d = d.dropna(subset=[x_col, y_col, z_metric]).reset_index(drop=True)

    # 聚合到每个 (x,y)
    g = d.groupby([x_col, y_col])[z_metric]
    z_agg = g.median() if agg == "median" else g.mean()

    summary = pd.DataFrame({
        x_col: z_agg.index.get_level_values(0),
        y_col: z_agg.index.get_level_values(1),
        "z": z_agg.values,
        "std": g.std(ddof=1).values,     # n=1 => NaN
        "n": g.size().values,
        "min": g.min().values,
        "max": g.max().values,
    }).sort_values([x_col, y_col]).reset_index(drop=True)

    if save_summary_csv is not None:
        summary.to_csv(save_summary_csv, index=False)

    def _plot_contour(long_df, z_col, title, cbar_label, annotate=False):
        _require_col(long_df, x_col)
        _require_col(long_df, y_col)
        _require_col(long_df, z_col)

        x = long_df[x_col].to_numpy()
        y = long_df[y_col].to_numpy()
        z = long_df[z_col].to_numpy()

        # axis transform
        xp = np.log10(x) if use_log_x else x
        yp = np.log10(y) if use_log_y else y

        xu = np.unique(xp)
        yu = np.unique(yp)
        full_grid = (len(xp) == len(xu) * len(yu))

        plt.figure()

        if full_grid:
            tmp = long_df.copy()
            tmp["_x"] = xp
            tmp["_y"] = yp
            Z = tmp.pivot(index="_y", columns="_x", values=z_col)
            XX, YY = np.meshgrid(Z.columns.to_numpy(), Z.index.to_numpy())
            ZZ = np.ma.masked_invalid(Z.to_numpy())
            cf = plt.contourf(XX, YY, ZZ, levels=levels)
            plt.colorbar(cf, label=cbar_label)
            plt.contour(XX, YY, ZZ, levels=levels, linewidths=0.5)
        else:
            cf = plt.tricontourf(xp, yp, z, levels=levels)
            plt.colorbar(cf, label=cbar_label)
            plt.tricontour(xp, yp, z, levels=levels, linewidths=0.5)

        if show_points:
            plt.scatter(xp, yp, s=18)

        if annotate:
            # best point in z
            if best_mode == "min":
                i = np.nanargmin(z)
            else:
                i = np.nanargmax(z)
            bx, by, bz = x[i], y[i], z[i]
            bxp = np.log10(bx) if use_log_x else bx
            byp = np.log10(by) if use_log_y else by
            plt.scatter([bxp], [byp], s=60, marker="x")
            plt.text(bxp, byp, f" best\n({bx:g},{by:g})\n{z_col}={bz:.4g}", fontsize=9)

        xlabel = x_col + (" (log10)" if use_log_x else "")
        ylabel = y_col + (" (log10)" if use_log_y else "")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()

    # 主图：z
    _plot_contour(
        summary,
        z_col="z",
        title=f"Contour: {z_metric} over ({x_col}, {y_col}) [{agg}]",
        cbar_label=z_metric,
        annotate=annotate_best,
    )
    if save_fig_prefix is not None:
        plt.savefig(f"{save_fig_prefix}_z.png", dpi=200)

    # 次图：std（可选）
    if make_std_plot and ("std" in summary.columns) and (not summary["std"].isna().all()):
        std_df = summary.dropna(subset=["std"]).copy()
        _plot_contour(
            std_df,
            z_col="std",
            title=f"Contour: std({z_metric}) over ({x_col}, {y_col})",
            cbar_label=f"std({z_metric})",
            annotate=False,
        )
        if save_fig_prefix is not None:
            plt.savefig(f"{save_fig_prefix}_std.png", dpi=200)

    return summary
