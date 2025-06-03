from logging import getLogger
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Tuple,
)

# Non-installed imports are caught outside of the module
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except ImportError:
    pass

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


_logger = getLogger("Plotter")


def get_model_names(df: "pd.DataFrame") -> List[str]:
    """Extract unique model names from the DataFrame and sort them."""
    return sorted(df["model_name"].unique())


def get_colors(df: "pd.DataFrame") -> Dict[str, Tuple[float, float, float]]:
    """
    Generate a color map for the given model names.
    Uses seaborn's color palette to ensure distinct colors.
    """
    model_names: List[str] = get_model_names(df)
    fallback_palette = sns.color_palette(n_colors=len(model_names))
    return dict(zip(model_names, fallback_palette, strict=False))


def _setup_figure_and_save(
    out_folder: Path, filename: str, figsize: Tuple[int, int] | None = None
) -> Tuple["Figure", str]:
    """Setup figure with optional custom size and return save path."""
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    return fig, str(out_folder / filename)


def _save_and_close_plot(save_path: str, dpi: int = 300) -> None:
    """Save plot to file and close it."""
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Plot saved as {save_path}")
    plt.close()


def _prepare_categorical_data(
    df: "pd.DataFrame", models: List[str], features: List[str]
) -> "pd.DataFrame":
    """Prepare melted DataFrame with categorical ordering for plotting."""
    # Melt so each row is (model_name, publication_uuid, Feature, Value)
    melted_pd = pd.melt(
        df,
        id_vars=["model_name", "publication_uuid"],
        value_vars=features,
        var_name="Feature",
        value_name="Value",
    )

    # Ensure categorical order for plotting
    melted_pd["model_name"] = pd.Categorical(
        melted_pd["model_name"], categories=models, ordered=True
    )
    melted_pd["Feature"] = pd.Categorical(
        melted_pd["Feature"], categories=features, ordered=True
    )

    return melted_pd


def _apply_model_colors_to_xticklabels(
    ax: "Axes", color_map: Dict[str, Tuple[float, float, float]]
) -> None:
    """Apply model colors to x-axis tick labels."""
    for tick_label in ax.get_xticklabels():
        label_text = tick_label.get_text()
        if label_text in color_map:
            tick_label.set_color(color_map[label_text])


def plot_all_features(df: "pd.DataFrame", out_folder: Path) -> None:
    """Creates one huge plot for all evaluated features across all models. Only meant
    for exploratory analysis, not for publication.
    """
    print("Plotting feature distributions...")

    color_map = get_colors(df)
    models = get_model_names(df)

    # Exclude non-numeric or identifier columns
    id_cols = ["model_name", "publication_uuid"]
    feature_cols = [
        col
        for col in df.columns
        if col not in id_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    features = list(feature_cols)

    melted_pd = _prepare_categorical_data(df, models, features)

    g = sns.catplot(
        data=melted_pd,
        x="model_name",
        y="Value",
        col="Feature",
        palette=color_map,
        hue="model_name",  # Color violins by model_name
        kind="violin",
        col_wrap=4,
        sharey=False,
        sharex=False,
        height=6,
        aspect=1.5,
        legend_out=True,  # Place legend outside the plot area,
        alpha=0.5,
    )

    # Simplify subplot titles to just the feature name
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=45, ha="right")

    # Add individual data points with strip plot on each subplot
    for ax in g.axes.flat:
        # Get the feature for this subplot
        feature = ax.get_title()

        # Filter data for this feature and plot individual points
        feature_data = melted_pd[melted_pd["Feature"] == feature]

        sns.stripplot(
            x="model_name",
            y="Value",
            hue="model_name",
            data=feature_data,
            ax=ax,
            size=3,
            alpha=0.8,
            jitter=0.3,
            edgecolor="black",
            linewidth=0.6,
            palette=color_map,
        )

        _apply_model_colors_to_xticklabels(ax, color_map)

    # Further adjust bottom to make space for rotated x-tick labels
    g.tight_layout()
    plt.subplots_adjust(bottom=0.3, wspace=0.3, hspace=0.6)

    save_path = str(out_folder / "feature_distributions.png")
    _save_and_close_plot(save_path)


def plot_pubtator_evaluation(df: "pd.DataFrame", out_folder: Path) -> None:
    """
    Create a stacked bar chart: one bar per publication, grouped by model_name.
    Each bar is stacked with Pubtator entities, positive predictions and false positives.
    Uses a broken y-axis to show outliers while focusing on the main range.
    """
    from matplotlib.patches import Patch

    print("Plotting PubTator evaluation...")

    color_map = get_colors(df)
    models = get_model_names(df)
    pubs = df["publication_uuid"].unique()
    n_pubs = len(pubs)
    bar_width = 0.12
    gap = 0.15  # gap between model groups

    # Prepare data for plotting
    bar_positions = []
    bar_colors = []
    for i in range(len(models)):
        for j in range(n_pubs):
            bar_positions.append(i * (n_pubs * bar_width + gap) + j * bar_width)
            bar_colors.append(color_map[models[i]])

    # Get values in the same order as bar_positions
    fp_vals = []
    positive_vals = []
    total_vals = []
    for model in models:
        for pub in pubs:
            row = df[(df["model_name"] == model) & (df["publication_uuid"] == pub)]
            if not row.empty:
                fp = int(row["false_positive_pubtator_entities"].iloc[0])
                positive = int(row["true_positive_pubtator_entities"].iloc[0]) + fp
                total = int(row["pubtator_list"].iloc[0])
            else:
                fp = 0
                positive = 0
                total = 0
            fp_vals.append(fp)
            positive_vals.append(positive)
            total_vals.append(total)
    positive_remainder = [
        max(positive - fp, 0)
        for positive, fp in zip(positive_vals, fp_vals, strict=True)
    ]
    total_remainder = [
        max(total - pos, 0)
        for total, pos in zip(total_vals, positive_remainder, strict=True)
    ]

    y_main_max = 50
    y_outlier_min = 50
    y_outlier_max = max(total_vals) + 10

    fig, (ax_out, ax_main) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(max(10, 1.25 * len(models)), 6),
        gridspec_kw={"height_ratios": [1, 3]},
    )
    fig.subplots_adjust(hspace=0.05)

    # Plot bars on both axes
    for ax in [ax_out, ax_main]:
        ax.bar(
            bar_positions,
            fp_vals,
            bar_width,
            color=bar_colors,
            alpha=0.8,
            label="False Positives" if ax is ax_main else None,
            zorder=2,
            edgecolor="#444",
            hatch="///",
            linewidth=1.2,
        )
        ax.bar(
            bar_positions,
            positive_remainder,
            bar_width,
            bottom=fp_vals,
            color=bar_colors,
            alpha=0.5,
            label="Positive Pubtator Entity Count" if ax is ax_main else None,
            zorder=2,
            edgecolor=bar_colors,
        )
        ax.bar(
            bar_positions,
            total_remainder,
            bar_width,
            bottom=positive_vals,
            color="white",
            alpha=0.9,
            label="Total Pubtator entity count" if ax is ax_main else None,
            zorder=2,
            edgecolor=bar_colors,
        )

    # Set y-limits (main plot below, outlier above)
    ax_main.set_ylim(0, y_main_max)
    ax_out.set_ylim(y_outlier_min, y_outlier_max)

    # Hide spines between axes
    ax_main.spines["top"].set_visible(False)
    ax_out.spines["bottom"].set_visible(False)
    ax_out.tick_params(labeltop=False)  # don't put tick labels at the top

    # Remove ticks and labels from outlier axis
    ax_out.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    ax_out.set_xticks([])
    ax_out.set_xticklabels([])

    for ax in [ax_out]:
        y_min, y_max = ax.get_ylim()
        ticks = list(range(int(y_min), int(y_max) + 1, 100))
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.5, zorder=0)

    # indicate break
    ax_out.plot((0, 1), (0, 0), transform=ax_out.transAxes, color="k", clip_on=False)
    ax_main.plot((0, 1), (1, 1), transform=ax_main.transAxes, color="k", clip_on=False)

    # Set x-ticks in the middle of each model group
    ax_main.set_xticks(
        [
            i * (n_pubs * bar_width + gap) + (n_pubs / 2 - 0.5) * bar_width
            for i in range(len(models))
        ]
    )
    ax_main.set_xticklabels(models, rotation=45, ha="right", fontsize=13)

    # Remove x-ticks but keep ticklabels
    ax_main.tick_params(axis="x", which="both", bottom=False, top=False)

    # Color x-axis tick labels by model color
    _apply_model_colors_to_xticklabels(ax_main, color_map)

    ax_main.set_ylabel("Entities", fontsize=15)
    ax_out.set_title(
        "PubTator Entity Counts by Model and Publication",
        fontsize=17,
        pad=5,
    )
    legend_patches = [
        Patch(facecolor="white", edgecolor="#333", alpha=0.6, label="Total Entities"),
        Patch(facecolor="#333", alpha=0.3, label="Positive Entities"),
        Patch(
            facecolor="#333",
            alpha=0.8,
            label="False Positives",
            hatch="///",
            edgecolor="#444",
        ),
    ]
    ax_out.legend(
        handles=legend_patches,
        fontsize=11,
        frameon=True,
        loc="upper right",
        ncol=3,  # Make legend a single row
        columnspacing=1.2,
        handletextpad=0.7,
        borderaxespad=0.7,
    )

    # Adjust x-axis limits to remove extra whitespace
    plt.tight_layout(rect=(0, 0, 0.95, 1))
    leftmost = bar_positions[0] - gap
    rightmost = bar_positions[-1] + gap
    ax_main.set_xlim(leftmost, rightmost)
    ax_out.set_xlim(leftmost, rightmost)

    # Save the plot
    save_path = str(out_folder / "pubtator_evaluation.png")
    _save_and_close_plot(save_path)
