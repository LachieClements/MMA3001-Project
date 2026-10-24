"""Development and final figures generated exclusively from saved records."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airfoil_noise.data import file_sha256
from airfoil_noise.development import write_json


def plot_development(search_directory: str | Path, benchmark_directory: str | Path,
                     directory: str | Path) -> list[str]:
    """Plot saved accuracy, timing, complexity and sensitivity without fitting.

    Parameters
    ----------
    search_directory, benchmark_directory : str or pathlib.Path
        Completed, separate development search and repeated benchmark records.
    directory : str or pathlib.Path
        New figure directory; existing outputs are protected.

    Returns
    -------
    list of str
        Figure stems exported as PNG and SVG. The manifest binds source records.
        No final-test results or selected favourable examples are used.
    """
    search, benchmark, destination = map(Path, (search_directory, benchmark_directory, directory))
    if destination.exists():
        raise FileExistsError(destination)
    used = {}

    def read(path):
        used[str(path)] = file_sha256(path)
        return json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else pd.read_csv(path)

    frozen = read(search / "selected_configurations.json")
    timings = read(benchmark / "summary.csv")
    manifest = read(benchmark / "benchmark.json")
    if manifest["selection_sha256"] != file_sha256(search / "selected_configurations.json"):
        raise ValueError("Benchmark selection differs from the search.")
    variants = read(search / "variants.json")
    scores = read(search / "variant_scores.csv")
    configurations = read(search / "configuration_scores.csv")
    destination.mkdir(parents=True, exist_ok=False)
    figures = []

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(destination / f"{name}.{extension}", dpi=180)
        plt.close(fig)
        figures.append(name)

    def accuracy_time(ax, prefix, label):
        for _, row in timings.iterrows():
            if not row["complete"]:
                continue
            median = row[f"{prefix}_median"]
            error = np.array([[max(0, median - row[f"{prefix}_q25"])],
                              [max(0, row[f"{prefix}_q75"] - median)]]) * 1000
            ax.errorbar(median * 1000, row["representative_oof_equal_condition_mae_db"],
                        xerr=error, fmt="o", capsize=3)
            ax.annotate(row["model"], (median * 1000, row["representative_oof_equal_condition_mae_db"]),
                        xytext=(4, 5), textcoords="offset points", fontsize=9)
        ax.set(xscale="log", xlabel=label, ylabel="OOF equal-condition MAE (dB)")
        ax.margins(x=.25, y=.16)
        ax.grid(alpha=.2)

    fig, ax = plt.subplots(figsize=(7, 4))
    accuracy_time(ax, "fit_seconds", "Selected fit on all development data (ms; median/IQR)")
    ax.set_title("Development accuracy versus training time (MLP: seed 17)")
    save(fig, "accuracy_vs_training_time")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    accuracy_time(axes[0], "batch_seconds", "256-row batch predict (ms; median/IQR)")
    accuracy_time(axes[1], "single_seconds", "One-row predict call (ms; block means, median/IQR)")
    fig.suptitle("End-to-end prediction, including preprocessing (MLP: seed 17)")
    save(fig, "accuracy_vs_prediction_time")

    tree = scores[(scores["model"] == "tree") & scores["eligible"]]
    if not tree.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(tree["mean_structural_count"], tree["equal_condition_mae_db"], label="OOF validation")
        ax.scatter(tree["mean_structural_count"], tree["training_equal_condition_mae_db"], marker="x", label="Training (repeated fold-training rows)")
        specifications = {item["variant_id"]: item["specification"]["parameters"] for item in variants}
        labels = []
        for position, (_, row) in enumerate(tree.iterrows(), start=1):
            params = specifications[row["variant_id"]]
            labels.append(f'T{position}: depth={params["max_depth"]}, leaf={params["min_samples_leaf"]}')
            # Shallow trees can coincide; short labels plus a key stay readable.
            offset = (4, 10 + 13 * (position - 1)) if position <= 3 else (4, 7)
            ax.annotate(f"T{position}",
                        (row["mean_structural_count"], row["equal_condition_mae_db"]),
                        xytext=offset, textcoords="offset points", fontsize=8)
        ax.text(1.03, .55, "\n".join(labels), transform=ax.transAxes, fontsize=8, va="top")
        ax.margins(x=.15, y=.2)
        ax.set(xscale="log", xlabel="Mean fitted tree node count across folds", ylabel="Equal-condition MAE (dB)")
        ax.legend(fontsize=8)
        save(fig, "tree_complexity")

    svr = scores[(scores["model"] == "svr") & scores["eligible"]].copy()
    if not svr.empty:
        specifications = {item["variant_id"]: item["specification"]["parameters"] for item in variants}
        for name in ("C", "epsilon", "gamma"):
            svr[name] = [specifications[key][name] for key in svr["variant_id"]]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for (epsilon, gamma), rows in svr.groupby(["epsilon", "gamma"]):
            rows = rows.sort_values("C")
            axes[0].plot(rows["C"], rows["equal_condition_mae_db"], marker="o", label=f"eps={epsilon}, gamma={gamma}")
        axes[0].set(xscale="log", xlabel="C", ylabel="OOF equal-condition MAE (dB)")
        axes[0].legend(fontsize=7)
        axes[1].scatter(svr["mean_structural_count"], svr["equal_condition_mae_db"])
        axes[1].set(xlabel="Mean support-vector count", ylabel="OOF equal-condition MAE (dB)")
        save(fig, "svr_sensitivity")

    mlp = scores[(scores["model"] == "mlp") & scores["eligible"]]
    if not mlp.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        labels = []
        for position, (config_id, group) in enumerate(mlp.groupby("configuration_id", sort=False)):
            spec = next(item for item in variants if item["configuration_id"] == config_id)["specification"]["parameters"]
            labels.append(f'{tuple(spec["hidden_layer_sizes"])}\na={spec["alpha"]}')
            for offset, (_, row) in zip([-.15, 0, .15], group.iterrows()):
                ax.scatter(position + offset, row["equal_condition_mae_db"])
                ax.annotate(str(int(row["seed"])), (position + offset, row["equal_condition_mae_db"]),
                            xytext=(2, 3), textcoords="offset points", fontsize=7)
            mean = configurations.set_index("configuration_id").loc[config_id, "selection_score_db"]
            ax.plot([position - .22, position + .22], [mean, mean], color="black")
        ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel="OOF equal-condition MAE (dB)",
               title="MLP seed sensitivity (labels: seed; black line: seed mean)")
        save(fig, "mlp_sensitivity")

    chosen = frozen["selected"].get("mlp")
    if chosen:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for item in variants:
            if item["configuration_id"] != chosen["configuration_id"]:
                continue
            diagnostics = read(search / "runs" / item["variant_id"] / "diagnostics.json")
            for record in diagnostics:
                if record["status"] != "success":
                    continue
                fitted = record["fitted"]["steps"]["regressor"]
                axes[0].plot(np.arange(1, len(fitted["loss_curve_"]) + 1), fitted["loss_curve_"],
                             alpha=.6, linewidth=.8, label=f'{item["seed"]}/f{record["fold"]}' + (" W" if record["convergence_warning"] else ""))
        axes[0].set(yscale="log", xlabel="Training epoch", ylabel="Training loss (sklearn squared-error loss)",
                    title="Selected MLP: all 15 seed/fold curves; W = warning")
        axes[0].legend(fontsize=6, ncol=3)
        for seed, group in mlp.groupby("seed"):
            axes[1].plot(group["configuration_id"], group["mean_iterations"], marker="o", label=f"seed {int(seed)}")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].set(ylabel="Mean completed epochs across folds", title="All MLP configurations")
        axes[1].legend()
        save(fig, "mlp_convergence")

    condition_tables = []
    for model, chosen in frozen["selected"].items():
        table = read(search / "runs" / chosen["representative_variant_id"] / "condition_metrics.csv")
        table["method"] = model
        condition_tables.append(table)
    if condition_tables:
        table = pd.concat(condition_tables)
        identifiers = sorted(table["condition_id"].unique())
        fig, ax = plt.subplots(figsize=(12, 4))
        for model, rows in table.groupby("method", sort=False):
            rows = rows.set_index("condition_id").loc[identifiers]
            ax.plot(range(1, len(rows) + 1), rows["mae_db"], linewidth=.8, label=model)
        ax.set(xlabel="All development conditions, ordered by condition ID (see saved table)",
               ylabel="OOF condition MAE (dB)", title="Every development condition; MLP uses seed 17")
        ax.legend(ncol=5, fontsize=8)
        save(fig, "condition_errors")
    write_json(destination / "manifest.json", {"kind": "development_figures", "source_sha256": used,
                                               "plotting_source_sha256": file_sha256(Path(__file__)),
                                               "figures": figures, "fits_attempted": 0, "final_test_evaluated": False})
    return figures


def plot_final(run_directory: str | Path, directory: str | Path) -> list[str]:
    """Plot every saved holdout observation/condition without model calls.

    Parameters
    ----------
    run_directory : str or pathlib.Path
        Complete records validated by final_evaluation.load_final_records.
    directory : str or pathlib.Path
        New protected figure directory.

    Returns
    -------
    list of str
        PNG/SVG stems: five-model parity and residual panels, all-condition
        error heatmap, and frequency sweeps paginated in condition-ID order.
        No examples are selected by prediction errors.
    """
    from airfoil_noise.final_evaluation import MODEL_ORDER, load_final_records
    source, destination = Path(run_directory), Path(directory)
    if destination.exists():
        raise FileExistsError(destination)
    records = load_final_records(source)
    predictions, inputs = records["predictions"], records["inputs"]
    destination.mkdir(parents=True, exist_ok=False)
    figures = []

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(destination / f"{name}.{extension}", dpi=180)
        plt.close(fig)
        figures.append(name)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharex=True, sharey=True)
    lower = min(predictions["truth_db"].min(), predictions["prediction_db"].min())
    upper = max(predictions["truth_db"].max(), predictions["prediction_db"].max())
    margin = max(.5, .04 * (upper - lower))
    for name, ax in zip(MODEL_ORDER, axes):
        rows = predictions[predictions["model"] == name]
        ax.scatter(rows["truth_db"], rows["prediction_db"], s=10, alpha=.6)
        ax.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "k--", linewidth=.8)
        ax.set(title=name + (" (primary)" if name == "svr" else ""), xlabel="Measured (dB)",
               xlim=(lower - margin, upper + margin), ylim=(lower - margin, upper + margin))
    axes[0].set_ylabel("Predicted (dB)")
    save(fig, "predicted_vs_measured")

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharex=True, sharey=True)
    joined = predictions.merge(inputs[["row_id", "frequency_hz"]], on="row_id", validate="many_to_one")
    for name, ax in zip(MODEL_ORDER, axes):
        rows = joined[joined["model"] == name]
        ax.scatter(rows["frequency_hz"], rows["truth_db"] - rows["prediction_db"], s=10, alpha=.6)
        ax.axhline(0, color="black", linestyle="--", linewidth=.8)
        ax.set(xscale="log", xlabel="Frequency (Hz)", title=name)
    axes[0].set_ylabel("Measured minus predicted (dB)")
    save(fig, "residuals_by_frequency")

    identifiers = sorted(inputs["condition_id"].unique())
    table = records["condition_metrics"].pivot(index="model", columns="condition_id", values="mae_db")
    table = table.loc[list(MODEL_ORDER), identifiers]
    fig, ax = plt.subplots(figsize=(max(7, .45 * len(identifiers)), 4))
    grid = ax.imshow(table, aspect="auto", cmap="viridis")
    ax.set(yticks=range(5), yticklabels=MODEL_ORDER, xticks=range(len(identifiers)),
           xticklabels=[str(i + 1) for i in range(len(identifiers))],
           xlabel="All conditions, numbered in sorted condition-ID order (key in condition_key.csv)",
           title="Per-condition MAE (dB); every withheld condition")
    fig.colorbar(grid, ax=ax, label="MAE (dB)")
    save(fig, "condition_mae")
    from airfoil_noise.development import write_table
    write_table(destination / "condition_key.csv", pd.DataFrame({
        "condition_number": range(1, len(identifiers) + 1), "condition_id": identifiers,
    }))

    # Every condition appears once, on pages of at most eleven panels.
    for start in range(0, len(identifiers), 11):
        page = identifiers[start:start + 11]
        fig, axes = plt.subplots((len(page) + 2) // 3, 3, figsize=(15, 3 * ((len(page) + 2) // 3)),
                                 squeeze=False)
        for condition, ax in zip(page, axes.flat):
            rows = joined[joined["condition_id"] == condition].sort_values("frequency_hz")
            operating = inputs[inputs["condition_id"] == condition].iloc[0]
            title = (
                f'{identifiers.index(condition) + 1}: c={operating["chord_m"]:.4g} m, '
                f'v={operating["velocity_m_s"]:.4g} m/s\n'
                f'angle={operating["angle_deg"]:.4g} deg'
            )
            measured = rows[rows["model"] == "training_mean"]
            ax.plot(measured["frequency_hz"], measured["truth_db"], "k.-", label="measured")
            for name in MODEL_ORDER:
                subset = rows[rows["model"] == name]
                ax.plot(subset["frequency_hz"], subset["prediction_db"], linewidth=.9, label=name)
            ax.set(xscale="log", xlabel="Frequency (Hz)", ylabel="Noise (dB)", title=title)
        for ax in list(axes.flat)[len(page):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=7, ncol=2)
        save(fig, f"frequency_sweeps_{start // 11 + 1:02d}")
    write_json(destination / "manifest.json", {
        "kind": "saved_final_prediction_figures", "run_kind": records["run"]["kind"],
        "source_prediction_sha256": file_sha256(source / "predictions.csv"),
        "evaluation_sha256": file_sha256(source / "evaluation.json"),
        "plotting_source_sha256": file_sha256(Path(__file__)),
        "figures": figures, "condition_ids_in_order": identifiers,
        "fits_attempted": 0, "prediction_calls": 0,
    })
    return figures
