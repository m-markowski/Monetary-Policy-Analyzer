import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Shared visual baseline
TEMPLATE = "plotly_white"


def distribution_plot(
    series: pd.Series,
    bins: int = 50,
    show_kde: bool = True,
    show_normal: bool = True,
    title: str | None = None,
) -> go.Figure | None:
    """
    Histogram (counts) of a numeric variable with optional KDE and normal overlays.

    Bins are pinned to exactly `bins` equal-width intervals across the data range.
    The KDE and fitted-normal densities are scaled by n * bin_width so they overlay
    correctly on the count axis; the gap between them shows how far the variable
    departs from normality.

    Args:
        series (pd.Series): Numeric variable to plot.
        bins (int): Number of equal-width histogram bins.
        show_kde (bool): Overlay a kernel density estimate (scaled to counts).
        show_normal (bool): Overlay a fitted normal density (scaled to counts).
        title (str | None): Figure title; a default is derived from the name.

    Returns:
        go.Figure | None: The figure, or None if the series has no data.
    """
    values = series.dropna().to_numpy()
    n = values.size
    if n == 0:
        return None

    name = series.name or "value"
    start, end = float(values.min()), float(values.max())
    width = (end - start) / bins if end > start else 0.0

    fig = go.Figure()
    if width > 0:
        edges = np.linspace(start, end, bins + 1)
        counts, _ = np.histogram(values, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2
        bin_bounds = np.column_stack([edges[:-1], edges[1:]])
        fig.add_bar(
            x=centers,
            y=counts,
            width=width * 0.98,
            name="Histogram",
            opacity=0.6,
            customdata=bin_bounds,
            hovertemplate=(
                "Bin start: %{customdata[0]:.2f}<br>Bin end: %{customdata[1]:.2f}<br>Count: %{y}<extra></extra>"
            ),
        )
    else:
        fig.add_histogram(x=values, name="Histogram", opacity=0.6)

    if (show_kde or show_normal) and width > 0:
        grid = np.linspace(start, end, 200)
        scale = n * width  # density integrates to 1 -> expected counts per bin
        if show_kde and n > 1:
            kde = stats.gaussian_kde(values)
            fig.add_scatter(
                x=grid,
                y=kde(grid) * scale,
                mode="lines",
                name="KDE",
                hovertemplate="Value: %{x:.2f}<br>Est. count: %{y:.2f}<extra>KDE</extra>",
            )
        if show_normal:
            mu, sigma = values.mean(), values.std(ddof=1)
            if sigma > 0:
                fig.add_scatter(
                    x=grid,
                    y=stats.norm.pdf(grid, mu, sigma) * scale,
                    mode="lines",
                    name="Normal fit",
                    line={"dash": "dash"},
                    hovertemplate="Value: %{x:.2f}<br>Est. count: %{y:.2f}<extra>Normal fit</extra>",
                )

    fig.update_layout(
        template=TEMPLATE,
        title=title or f"Distribution - {name}",
        xaxis_title=name,
        yaxis_title="Count",
    )
    return fig


def probability_plot(series: pd.Series, kind: str = "qq", title: str | None = None) -> go.Figure | None:
    """
    Quantile-quantile or probability-probability plot against the normal.

    A QQ plot compares sample quantiles with theoretical normal quantiles; a PP
    plot compares the empirical CDF with the fitted normal CDF. Points on the
    reference line indicate normality.

    Args:
        series (pd.Series): Numeric variable to assess.
        kind (str): 'qq' or 'pp'.
        title (str | None): Figure title; a default is derived from the name.

    Returns:
        go.Figure | None: The figure, or None for fewer than three points, a
        zero-variance series, or an unknown `kind`.
    """
    values = np.sort(series.dropna().to_numpy())
    n = values.size
    if n < 3:
        return None
    mu, sigma = values.mean(), values.std(ddof=1)
    if sigma == 0:
        return None

    name = series.name or "value"
    fig = go.Figure()
    if kind == "qq":
        (osm, osr), (slope, intercept, _) = stats.probplot(values, dist="norm")
        fig.add_scatter(
            x=osm,
            y=osr,
            mode="markers",
            name="Sample",
            hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>",
        )
        fig.add_scatter(
            x=osm,
            y=slope * osm + intercept,
            mode="lines",
            name="Reference",
            line={"dash": "dash"},
            hoverinfo="skip",
        )
        x_title, y_title = "Theoretical quantiles", "Sample quantiles"
    elif kind == "pp":
        empirical = (np.arange(1, n + 1) - 0.5) / n
        theoretical = stats.norm.cdf(values, mu, sigma)
        fig.add_scatter(
            x=theoretical,
            y=empirical,
            mode="markers",
            name="Sample",
            hovertemplate="Theoretical CDF: %{x:.2f}<br>Empirical CDF: %{y:.2f}<extra></extra>",
        )
        fig.add_scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Reference",
            line={"dash": "dash"},
            hoverinfo="skip",
        )
        x_title, y_title = "Theoretical CDF", "Empirical CDF"
    else:
        return None

    fig.update_layout(
        template=TEMPLATE,
        title=title or f"{kind.upper()} plot - {name}",
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig


def box_or_violin(
    series: pd.Series,
    groups: pd.Series | None = None,
    kind: str = "box",
    title: str | None = None,
) -> go.Figure | None:
    """
    Box or violin plot of a variable, optionally split by group.

    With no groups it draws a single distribution; with a label series it draws
    one box/violin per group, which is the visual companion to `compare_groups`.

    Args:
        series (pd.Series): Numeric variable to plot.
        groups (pd.Series | None): Group labels index-aligned with `series`.
        kind (str): 'box' or 'violin'.
        title (str | None): Figure title; a default is derived from the name.

    Returns:
        go.Figure | None: The figure, or None if no data remains after dropping
        missing values.
    """
    frame = pd.DataFrame({"value": series})
    if groups is not None:
        frame["group"] = groups
    frame = frame.dropna()
    if frame.empty:
        return None

    name = series.name or "value"
    fig = go.Figure()
    if groups is None:
        if kind == "violin":
            fig.add_trace(go.Violin(y=frame["value"], name=name, box_visible=True, meanline_visible=True))
        else:
            fig.add_trace(go.Box(y=frame["value"], name=name))
    else:
        for label, sub in frame.groupby("group", observed=True):
            if kind == "violin":
                fig.add_trace(go.Violin(y=sub["value"], name=str(label), box_visible=True, meanline_visible=True))
            else:
                fig.add_trace(go.Box(y=sub["value"], name=str(label)))

    if groups is None:
        median = frame["value"].median()
        fig.add_hline(y=median, line_dash="dash", line_color="red", line_width=2, opacity=0.8)
        fig.add_annotation(
            x=0.02,
            y=median,
            xref="paper",
            yref="y",
            text=f"Median: {median:.2f}",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"color": "red", "size": 13.5},
        )

    fig.update_layout(
        template=TEMPLATE,
        title=title or f"{kind.capitalize()} - {name}",
        yaxis_title="",
        xaxis_title="",
        showlegend=groups is not None,
    )
    if groups is None:
        fig.update_xaxes(showticklabels=False)

    fig.update_yaxes(hoverformat=".2f")

    return fig


def grouped_histogram(
    series: pd.Series,
    groups: pd.Series,
    bins: int = 30,
    title: str | None = None,
    color_map: dict[str, str] | None = None,
    density: bool = False,
) -> go.Figure | None:
    """
    Overlaid step histograms of a variable split into subgroups.

    Each group is drawn as an outline (step) rather than a filled bar, so no group
    is hidden behind another where the distributions overlap. Densities can be
    shown instead of counts to compare the shape of groups of very different size.

    Args:
        series (pd.Series): Numeric variable to plot.
        groups (pd.Series): Subgroup labels index-aligned with `series`.
        bins (int): Number of histogram bins on the shared range.
        title (str | None): Figure title; a default is derived from the name.
        color_map (dict[str, str] | None): Optional label - colour mapping.
        density (bool): Plot per-group densities instead of raw counts.

    Returns:
        go.Figure | None: The figure, or None if fewer than two subgroups remain.
    """
    frame = pd.DataFrame({"value": series, "group": groups}).dropna()
    if frame["group"].nunique() < 2:
        return None

    name = series.name or "value"
    values = frame["value"].to_numpy()
    start, end = float(values.min()), float(values.max())
    if end <= start:
        return None

    edges = np.linspace(start, end, bins + 1)
    step_x = np.repeat(edges, 2)[1:-1]  # staircase: paired bin edges

    fig = go.Figure()
    for label, sub in frame.groupby("group", observed=True):
        counts, _ = np.histogram(sub["value"].to_numpy(), bins=edges, density=density)
        step_y = np.repeat(counts, 2)
        label = str(label)
        line = {"width": 2}
        if color_map:
            line["color"] = color_map.get(label)
        fig.add_scatter(
            x=step_x,
            y=step_y,
            name=label,
            mode="lines",
            line=line,
            hovertemplate=(
                f"Value: %{{x:.2f}}<br>{'Density' if density else 'Count'}: "
                f"%{{y:{'.3f' if density else '.0f'}}}<extra>{label}</extra>"
            ),
        )

    fig.update_layout(
        template=TEMPLATE,
        title=title or f"Distribution by group - {name}",
        xaxis_title=name,
        yaxis_title="Density" if density else "Count",
    )
    return fig


def category_counts(
    series: pd.Series,
    kind: str = "bar",
    title: str | None = None,
    color_map: dict[str, str] | None = None,
) -> go.Figure | None:
    """
    Bar or pie chart of category frequencies.

    Bar shows counts labelled above each bar; pie shows the share of total on each
    slice. Hover is disabled on both since the labels already carry the value. An
    optional label - colour mapping keeps the categories consistent with the other
    charts in the same view.

    Args:
        series (pd.Series): Categorical variable to count.
        kind (str): 'bar' or 'pie'.
        title (str | None): Figure title; a default is derived from the name.
        color_map (dict[str, str] | None): Optional label - colour mapping.

    Returns:
        go.Figure | None: The figure, or None if the variable is all-missing.
    """
    counts = series.dropna().value_counts()
    if counts.empty:
        return None

    name = series.name or "category"
    labels = counts.index.astype(str)
    colors = [color_map.get(lbl) for lbl in labels] if color_map else None

    if kind == "pie":
        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=counts.to_numpy(),
                hole=0.3,
                sort=False,
                marker={"colors": colors},
                textinfo="percent",
                textfont={"size": 16, "color": "white"},
                hoverinfo="skip",
            )
        )
    else:
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=counts.to_numpy(),
                marker={"color": colors},
                text=counts.to_numpy(),
                textposition="outside",
                textfont={"size": 14},
                cliponaxis=False,
                hoverinfo="skip",
            )
        )
        fig.update_layout(xaxis_title=name, yaxis_title="Count")

    fig.update_layout(template=TEMPLATE, title=title or f"Counts - {name}")
    return fig


def scatter_ols(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    add_ols: bool = True,
    title: str | None = None,
    color_map: dict[str, str] | None = None,
) -> go.Figure | None:
    """
    Scatter plot of two numeric variables with an optional OLS fit.

    A single least-squares line is fitted across all points (its slope and
    correlation are shown in the legend); an optional categorical column colours
    the markers, e.g. by regime.

    Args:
        df (pd.DataFrame): Source dataset.
        x (str): Column for the x-axis.
        y (str): Column for the y-axis.
        color (str | None): Categorical column used to colour points.
        add_ols (bool): Draw the ordinary-least-squares line.
        title (str | None): Figure title; a default is derived from x and y.
        color_map (dict[str, str] | None): Optional label - colour mapping.

    Returns:
        go.Figure | None: The figure, or None if fewer than three complete rows
        remain.
    """
    cols = [x, y] + ([color] if color else [])
    data = df[cols].dropna()
    if data.shape[0] < 3:
        return None

    if color is not None:
        data[color] = data[color].astype(str)
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            opacity=0.75,
            color_discrete_map=color_map,
        )
    else:
        fig = px.scatter(data, x=x, y=y, opacity=0.6)

    if add_ols and data[x].nunique() > 1:
        slope, intercept, r, _, _ = stats.linregress(data[x], data[y])
        line_x = np.array([data[x].min(), data[x].max()])
        fig.add_scattergl(
            x=line_x,
            y=slope * line_x + intercept,
            mode="lines",
            name=f"OLS fit (r={r:.2f})",
            line={"color": "red", "dash": "dash", "width": 3},
        )

    fig.update_layout(template=TEMPLATE, title=title or f"{y} vs {x}")
    return fig


def matrix_heatmap(
    matrix: pd.DataFrame | None,
    colorscale: str = "RdBu",
    zmin: float = -1.0,
    zmax: float = 1.0,
    show_values: bool | None = None,
    cbar_title: str = "value",
    title: str | None = None,
    value_label: str = "value",
) -> go.Figure | None:
    """
    Heatmap of a square association matrix.

    Renders any precomputed matrix, so it serves both correlation matrices
    (Pearson/Spearman/Kendall, defaults below) and a Cramer's V matrix; switch
    `colorscale` to 'Blues' with `zmin=0, zmax=1` for the latter. Cell values are
    annotated automatically for small matrices.

    Args:
        matrix (pd.DataFrame | None): Square matrix to display.
        colorscale (str): Plotly colourscale name.
        zmin (float): Lower bound of the colour range.
        zmax (float): Upper bound of the colour range.
        show_values (bool | None): Force cell annotations on/off; None annotates
            only matrices with at most 15 rows.
        cbar_title (str): Colourbar label.
        title (str | None): Figure title.
        value_label (str): Label for the value axis.

    Returns:
        go.Figure | None: The figure, or None if the matrix is empty.
    """
    if matrix is None or matrix.empty:
        return None

    annotate = matrix.shape[0] <= 15 if show_values is None else show_values
    heat = go.Heatmap(
        z=matrix.to_numpy(),
        x=matrix.columns.astype(str),
        y=matrix.index.astype(str),
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar={"title": cbar_title},
    )
    heat.update(hovertemplate=f"%{{y}} vs. %{{x}}<br>{value_label}: %{{z:.2f}}<extra></extra>")
    if annotate:
        heat.update(
            text=np.round(matrix.to_numpy(), 2),
            texttemplate="%{text}",
            textfont={"size": 12.5, "color": "black"},
        )

    fig = go.Figure(heat)
    fig.update_layout(template=TEMPLATE, title=title)
    fig.update_yaxes(autorange="reversed")
    return fig


def compare_lines(
    df: pd.DataFrame,
    columns: list[str],
    title: str | None = None,
    zero_line: bool = False,
    opacity: float = 0.85,
) -> go.Figure | None:
    """
    Stacked line charts of up to four series sharing a date axis.

    Each series occupies its own panel with an independent y-axis.
    A zero reference line can be drawn on every panel, which is useful
    for returns and first-difference series.

    Args:
        df (pd.DataFrame): Source data, date-indexed.
        columns (list[str]): Up to four column names to plot.
        title (str | None): Figure title.
        zero_line (bool): Draw a dashed zero reference line on every panel.
        opacity (float): Line opacity (0-1).

    Returns:
        go.Figure | None: The figure, or None if no valid columns are found.
    """
    cols = [c for c in columns if c in df.columns][:4]
    if not cols:
        return None

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    n = len(cols)

    if n == 1:
        s = df[cols[0]].dropna()
        fig = go.Figure()
        fig.add_scatter(x=s.index, y=s, name=cols[0], line={"color": colors[0], "width": 1.5}, opacity=opacity)
        if zero_line:
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.25)")
        fig.update_layout(
            template=TEMPLATE,
            title={
                "text": title if title else cols[0],
                "x": 0.5,
                "xanchor": "center",
            },
            hovermode="x unified",
        )
    else:
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=max(0.04, 0.22 / n),
            subplot_titles=cols,
        )
        for i, col in enumerate(cols):
            s = df[col].dropna()
            fig.add_scatter(
                x=s.index,
                y=s,
                name=col,
                row=i + 1,
                col=1,
                line={"color": colors[i], "width": 1.5},
                opacity=opacity,
                showlegend=False,
            )
            if zero_line:
                fig.add_hline(
                    y=0,
                    line_dash="dot",
                    line_color="rgba(0,0,0,0.25)",
                    row=i + 1,
                    col=1,
                )
        fig.update_layout(
            template=TEMPLATE,
            title=title,
            hovermode="x unified",
            height=280 + 180 * (n - 1),
        )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.5,
        griddash="dash",
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.5,
        griddash="dash",
    )

    fig.update_traces(yhoverformat=".2f")

    return fig


def levels_with_overlay(
    levels: pd.DataFrame,
    overlay: pd.DataFrame,
    columns: list[str],
    overlay_label: str = "Log return (%)",
    title: str | None = None,
) -> go.Figure | None:
    """
    Stacked panels of each feature's level with an overlaid transform on a
    secondary axis.

    One panel per column (up to four). The level is drawn on the left y-axis and
    the overlay (e.g. log returns) on the right, since the two live on different
    scales, so the size of the moves can be read against the level in the same
    panel. Columns absent from `overlay` show the level only.

    Args:
        levels (pd.DataFrame): Level series per column, date-indexed.
        overlay (pd.DataFrame): Overlay series, a subset of the same columns/index.
        columns (list[str]): Up to four columns to plot, in order.
        overlay_label (str): Legend and axis label for the overlay.
        title (str | None): Figure title.

    Returns:
        go.Figure | None: The figure, or None if no valid columns are found.
    """
    cols = [c for c in columns if c in levels.columns][:4]
    if not cols:
        return None

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    overlay_color = "#FFD400"
    n = len(cols)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=max(0.05, 0.24 / n),
        subplot_titles=cols,
        specs=[[{"secondary_y": True}] for _ in range(n)],
    )
    overlay_shown = False
    for i, col in enumerate(cols):
        lvl = levels[col].dropna()
        fig.add_scatter(
            x=lvl.index,
            y=lvl,
            name="Level",
            legendgroup="level",
            showlegend=i == 0,
            line={"color": colors[i], "width": 1.5},
            row=i + 1,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Level",
            row=i + 1,
            col=1,
            secondary_y=False,
            showgrid=True,
            gridwidth=0.5,
            griddash="dash",
        )
        ret = overlay[col].dropna() if col in overlay.columns else None
        if ret is not None and not ret.empty:
            fig.add_scatter(
                x=ret.index,
                y=ret,
                name=overlay_label,
                legendgroup="overlay",
                showlegend=not overlay_shown,
                line={"color": overlay_color, "width": 1},
                opacity=0.3,
                row=i + 1,
                col=1,
                secondary_y=True,
            )
            fig.add_hline(
                y=0,
                line_dash="dot",
                line_color="rgba(0,0,0,0.25)",
                row=i + 1,
                col=1,
                secondary_y=True,
            )
            fig.update_yaxes(title_text=overlay_label, row=i + 1, col=1, secondary_y=True, showgrid=False)
            overlay_shown = True

    fig.update_xaxes(showgrid=True, gridwidth=0.5, griddash="dash")

    layout_kwargs = {
        "template": TEMPLATE,
        "title": title,
        "hovermode": "x unified",
    }
    if n > 1:
        layout_kwargs["height"] = 280 + 180 * (n - 1)

    fig.update_layout(**layout_kwargs)
    fig.update_traces(yhoverformat=".2f")
    return fig


def explained_variance_plot(evr: pd.Series, title: str | None = None) -> go.Figure | None:
    """
    Scree plot: per-component variance share with a cumulative overlay.

    Bars show the variance explained by each principal component; the line shows
    the running total, which shows how many components capture most of the
    variation.

    Args:
        evr (pd.Series): Explained-variance ratio per component, indexed by name.
        title (str | None): Figure title.

    Returns:
        go.Figure | None: The figure, or None if the series is empty.
    """
    if evr is None or evr.empty:
        return None

    names = evr.index.astype(str)
    fig = go.Figure()
    fig.add_bar(
        x=names,
        y=evr.to_numpy() * 100,
        name="Per component",
        opacity=0.6,
        hovertemplate="%{x}<br>Explained: %{y:.1f}%<extra></extra>",
    )
    cum = evr.cumsum().to_numpy() * 100
    fig.add_scatter(
        x=names,
        y=cum,
        name="Cumulative",
        mode="lines+markers+text",
        text=[f"{v:.0f}%" for v in cum],
        textposition="top center",
        textfont={"size": 11},
        cliponaxis=False,
        hovertemplate="%{x}<br>Cumulative: %{y:.1f}%<extra></extra>",
    )
    fig.update_layout(
        template=TEMPLATE,
        title=title or "Explained variance",
        xaxis_title="Principal component",
        yaxis_title="Variance explained (%)",
    )
    return fig


def cluster_selection_plot(
    sweep: pd.DataFrame, best_k: int | None = None, title: str | None = None
) -> go.Figure | None:
    """
    K-selection diagnostics: inertia elbow and silhouette across k.

    Two stacked panels share the k axis: inertia (lower is tighter, look for the
    elbow) on top and mean silhouette (higher is better separated) below. An
    optional dashed line marks the suggested k.

    Args:
        sweep (pd.DataFrame): Output of `structure.kmeans_sweep`, indexed by k with
            'inertia' and 'silhouette' columns.
        best_k (int | None): k to highlight with a vertical line.
        title (str | None): Figure title.

    Returns:
        go.Figure | None: The figure, or None if the sweep is empty.
    """
    if sweep is None or sweep.empty:
        return None

    ks = sweep.index.astype(int)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=["Inertia (elbow)", "Silhouette"],
    )
    fig.add_scatter(
        x=ks,
        y=sweep["inertia"],
        mode="lines+markers",
        name="Inertia",
        line={"color": "#4c78a8"},
        row=1,
        col=1,
        showlegend=False,
        hovertemplate="k = %{x} clusters<br>Inertia: %{y:,.0f}<extra></extra>",
    )
    fig.add_scatter(
        x=ks,
        y=sweep["silhouette"],
        mode="lines+markers",
        name="Silhouette",
        line={"color": "#54a24b"},
        row=2,
        col=1,
        showlegend=False,
        hovertemplate="k = %{x} clusters<br>Silhouette: %{y:.3f}<extra></extra>",
    )
    if best_k is not None:
        for r in (1, 2):
            fig.add_vline(x=best_k, line_dash="dot", line_color="rgba(0,0,0,0.35)", row=r, col=1)
    fig.update_xaxes(title_text="Number of clusters (k)", dtick=1, row=2, col=1)
    fig.update_layout(template=TEMPLATE, height=460, title=title or "")
    return fig


def projection_scatter(
    scores: pd.DataFrame,
    color: pd.Series,
    x: str = "PC1",
    y: str = "PC2",
    color_map: dict[str, str] | None = None,
    centroids: bool = True,
    title: str | None = None,
) -> go.Figure | None:
    """
    2D scatter of PCA scores coloured by a categorical label.

    Plots each date on the first two principal components and colours it by either
    a data-driven cluster or a rule-based regime, so the two views of structure can
    be compared. Optional centroids mark each group's average position.

    Args:
        scores (pd.DataFrame): PCA scores with at least the `x` and `y` columns.
        color (pd.Series): Categorical labels index-aligned with `scores`.
        x (str): Component on the x-axis.
        y (str): Component on the y-axis.
        color_map (dict[str, str] | None): Optional label - colour mapping.
        centroids (bool): Mark each group's mean position with an X.
        title (str | None): Figure title.

    Returns:
        go.Figure | None: The figure, or None if fewer than three complete rows.
    """
    data = scores[[x, y]].join(color.rename("label")).dropna()
    if data.shape[0] < 3:
        return None
    data["label"] = data["label"].astype(str)

    fig = px.scatter(data, x=x, y=y, color="label", opacity=0.7, color_discrete_map=color_map, render_mode="svg")
    if centroids:
        centers = data.groupby("label", observed=True)[[x, y]].mean()
        label_color = {t.name: t.marker.color for t in fig.data}
        fig.add_scatter(
            x=centers[x],
            y=centers[y],
            mode="markers",
            name="Centroids",
            marker={
                "symbol": "x",
                "size": 16,
                "color": [label_color.get(lbl) for lbl in centers.index],
                "line": {"width": 2.5, "color": "white"},
            },
            hovertemplate="Centroid<br>%{x:.2f}, %{y:.2f}<extra></extra>",
        )
    fig.update_layout(template=TEMPLATE, title=title, legend_title_text="")
    return fig


def roc_curves(curves: dict, title: str | None = None) -> go.Figure | None:
    """
    Per-class one-vs-rest ROC curves with a chance diagonal.

    Args:
        curves (dict): Output of `evaluate.roc_curve_data` (label -> fpr/tpr/auc).
        title (str | None): Figure title.

    Returns:
        go.Figure | None: The figure, or None if no curve is available.
    """
    if not curves:
        return None
    fig = go.Figure()
    for label, d in curves.items():
        thr = d.get("thresholds")
        hover = "FPR: %{x:.2f}<br>TPR: %{y:.2f}"
        if thr is not None:
            hover += "<br>Threshold: %{customdata:.2f}"
        fig.add_scatter(
            x=d["fpr"],
            y=d["tpr"],
            mode="lines",
            name=f"{label} (AUC={d['auc']:.2f})",
            customdata=thr,
            hovertemplate=hover + f"<extra>{label}</extra>",
        )
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Chance (AUC=0.50)",
        line={"dash": "dash", "color": "#e45756", "width": 2},
        hoverinfo="skip",
    )
    fig.update_layout(
        template=TEMPLATE,
        title=title or "ROC curve",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
    )
    return fig


def confusion_heatmap(cm: pd.DataFrame | None, title: str | None = None) -> go.Figure | None:
    """
    Confusion matrix of raw counts as an annotated heatmap (rows = actual, columns = predicted).

    Args:
        cm (pd.DataFrame | None): Confusion matrix from `classification_metrics`.
        title (str | None): Figure title.

    Returns:
        go.Figure | None: The figure, or None if the matrix is empty.
    """
    if cm is None or cm.empty:
        return None
    heat = go.Heatmap(
        z=cm.to_numpy(dtype=float),
        x=cm.columns.astype(str),
        y=cm.index.astype(str),
        colorscale=[[0.0, "#f5f5f5"], [1.0, "#FF4B4B"]],
        colorbar={"title": "Count"},
        text=cm.to_numpy(),
        texttemplate="%{text}",
        textfont={"size": 12.5, "color": "black"},
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    )
    fig = go.Figure(heat)
    fig.update_layout(
        template=TEMPLATE,
        title=title or "Confusion matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def importance_bar(
    importance: pd.Series | None,
    top_n: int = 15,
    title: str | None = None,
    value_label: str = "Importance (0-100)",
) -> go.Figure | None:
    """
    Horizontal bar chart of feature importance (native, permutation or group).

    Args:
        importance (pd.Series | None): Importance scores, largest first.
        top_n (int): Number of top features to show.
        title (str | None): Figure title.
        value_label (str): X-axis / hover label for the score.

    Returns:
        go.Figure | None: The figure, or None if there is nothing to plot.
    """
    if importance is None or importance.empty:
        return None
    top = importance.head(top_n).iloc[::-1]  # largest ends up at the top of a horizontal bar
    fig = go.Figure(
        go.Bar(
            x=top.to_numpy(),
            y=top.index.astype(str),
            orientation="h",
            marker={"color": "#4c78a8"},
            text=[f"{v:.0f}" for v in top.to_numpy()],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"%{{y}}<br>{value_label}: %{{x:.1f}}<extra></extra>",
        )
    )
    fig.update_layout(template=TEMPLATE, title=title or "", xaxis_title=value_label, yaxis_title="")
    return fig


def class_probability_bar(proba: pd.Series | None, title: str | None = None) -> go.Figure | None:
    """
    Vertical bar chart of predicted class probabilities for a single observation.

    Args:
        proba (pd.Series | None): Probability per class, indexed by class label.
        title (str | None): Figure title.

    Returns:
        go.Figure | None: The figure, or None if there is nothing to plot.
    """
    if proba is None or proba.empty:
        return None
    labels = proba.index.astype(str)
    values = proba.to_numpy(dtype=float)
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker={"color": "#4c78a8"},
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Class %{x}<br>Probability: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis={"range": [0, 1], "tickformat": ".0%"},
    )
    return fig


def probability_histogram(
    y_proba,
    labels,
    y_true=None,
    bins: int = 25,
    title: str | None = None,
) -> go.Figure | None:
    """
    Per-class predicted-probability histograms, split by true membership.

    One panel per class shows the distribution of the model's predicted probability
    for that class; when `y_true` is given, samples that truly belong to the class
    are separated from the rest, stacked in each bin, so good separation shows green piled near 1 and red near 0.

    Args:
        y_proba: Predicted probabilities (n_samples x n_classes).
        labels (list): Ordered label set matching the probability columns.
        y_true: Observed labels (optional; enables the positive/negative split).
        bins (int): Histogram bins per panel.
        title (str | None): Figure title.

    Returns:
        go.Figure | None: The figure, or None if there are no probabilities.
    """
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.size == 0:
        return None
    n = len(labels)
    fig = make_subplots(rows=1, cols=n, shared_yaxes=True, subplot_titles=[f"P({lab})" for lab in labels])
    y_true = np.asarray(y_true) if y_true is not None else None
    xbins = {"start": 0.0, "end": 1.0, "size": 1.0 / bins}
    for j, lab in enumerate(labels):
        col = y_proba[:, j]
        if y_true is not None:
            fig.add_histogram(
                x=col[y_true == lab],
                name="Belongs to this class",
                legendgroup="pos",
                showlegend=j == 0,
                marker_color="#54a24b",
                xbins=xbins,
                hovertemplate="Predicted P: %{x:.2f}<br>Months: %{y}<extra>Belongs</extra>",
                row=1,
                col=j + 1,
            )
            fig.add_histogram(
                x=col[y_true != lab],
                name="Other classes",
                legendgroup="neg",
                showlegend=j == 0,
                marker_color="#e45756",
                xbins=xbins,
                hovertemplate="Predicted P: %{x:.2f}<br>Months: %{y}<extra>Other</extra>",
                row=1,
                col=j + 1,
            )
        else:
            fig.add_histogram(
                x=col,
                showlegend=False,
                marker_color="#4c78a8",
                xbins=xbins,
                hovertemplate="Predicted P: %{x:.2f}<br>Months: %{y}<extra></extra>",
                row=1,
                col=j + 1,
            )
    fig.update_layout(
        template=TEMPLATE,
        title=title or "Predicted probability by class",
        barmode="stack",
        legend_title_text="",
    )
    fig.update_xaxes(title_text="Predicted probability", range=[0, 1])
    fig.update_yaxes(title_text="Count", col=1)
    return fig


def predicted_vs_actual(y_true, y_pred, index=None, title: str | None = None) -> go.Figure | None:
    """
    Actual vs predicted values over the sample (a path for the monthly frame).

    Args:
        y_true: Observed target values.
        y_pred: Predicted target values.
        index: X-axis values (dates); a positional range if None.
        title (str | None): Figure title.

    Returns:
        go.Figure | None: The figure, or None if there is no data.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return None
    x = index if index is not None else np.arange(len(y_true))
    # Markers help on a short test window but clutter a 400-month span
    mode = "lines+markers" if y_true.size <= 60 else "lines"
    fig = go.Figure()
    fig.add_scatter(x=x, y=y_true, mode=mode, name="Actual", line={"color": "#1f77b4"})
    fig.add_scatter(x=x, y=y_pred, mode=mode, name="Predicted", line={"color": "#d62728", "dash": "dash"})
    fig.update_layout(template=TEMPLATE, title=title or "Predicted vs actual", yaxis_title="")
    fig.update_traces(yhoverformat=".2f")
    return fig


def residual_plot(y_true, y_pred, index=None, title: str | None = None) -> go.Figure | None:
    """
    Residuals (actual - predicted) over the sample with a zero reference line.

    Args:
        y_true: Observed target values.
        y_pred: Predicted target values.
        index: X-axis values (dates); a positional range if None.
        title (str | None): Figure title.

    Returns:
        go.Figure | None: The figure, or None if there is no data.
    """
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if resid.size == 0:
        return None
    x = index if index is not None else np.arange(len(resid))
    fig = go.Figure()
    fig.add_scatter(x=x, y=resid, mode="markers", name="Residual", marker={"color": "#4c78a8"})
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.4)")
    fig.update_layout(template=TEMPLATE, title=title or "Residuals (actual - predicted)", yaxis_title="Residual")
    fig.update_traces(yhoverformat=".2f")
    return fig


def cooks_distance_plot(cooks: pd.Series | None, threshold: float, title: str | None = None) -> go.Figure | None:
    """
    Cook's distance per observation with the 4/n rule-of-thumb threshold line.

    Args:
        cooks (pd.Series | None): Cook's distance per observation, date-indexed.
        threshold (float): The 4/n flagging threshold.
        title (str | None): Figure title.

    Returns:
        go.Figure | None: The figure, or None if there is no data.
    """
    if cooks is None or len(cooks) == 0:
        return None
    fig = go.Figure(
        go.Bar(
            x=cooks.index,
            y=cooks.to_numpy(),
            marker={"color": "#4c78a8"},
            hovertemplate="%{x|%Y-%m}<br>Cook's distance: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#d62728",
        annotation_text="4/n threshold",
        annotation_position="top left",
    )
    fig.update_layout(
        template=TEMPLATE, title=title or "Cook's distance by month", yaxis_title="Cook's distance", xaxis_title=""
    )
    return fig


def forecast_fan(
    history: pd.Series | None,
    forecast: dict | None,
    title: str | None = None,
    name: str = "Series",
) -> go.Figure | None:
    """
    History line plus a forecast mean and a shaded widening confidence band.

    Args:
        history (pd.Series | None): Recent observed values, date-indexed.
        forecast (dict | None): 'mean'/'lower'/'upper' Series over the forecast
            index (from `arima_forecast`).
        title (str | None): Figure title.
        name (str): Series name used in the default title.

    Returns:
        go.Figure | None: The figure, or None if there is no forecast.
    """
    if forecast is None:
        return None
    mean = forecast["mean"]
    if mean is None or len(mean) == 0:
        return None
    lower, upper = forecast["lower"], forecast["upper"]
    idx = list(mean.index)
    mean_x, mean_y = idx, list(np.asarray(mean))
    fig = go.Figure()
    h = pd.Series(dtype=float) if history is None else pd.Series(history).dropna()
    if not h.empty:
        fig.add_scatter(x=h.index, y=h.to_numpy(), mode="lines", name="History", line={"color": "#1f77b4"})
        mean_x = [h.index[-1], *idx]
        mean_y = [float(h.iloc[-1]), *mean_y]
    fig.add_scatter(
        x=idx + idx[::-1],
        y=list(np.asarray(upper)) + list(np.asarray(lower)[::-1]),
        fill="toself",
        fillcolor="rgba(214,39,40,0.15)",
        line={"color": "rgba(255,255,255,0)"},
        name="Prediction interval",
        hoverinfo="skip",
    )
    fig.add_scatter(x=mean_x, y=mean_y, mode="lines", name="Forecast", line={"color": "#d62728", "dash": "dash"})
    fig.update_layout(template=TEMPLATE, title=title or f"Forecast - {name}", yaxis_title="")
    fig.update_traces(yhoverformat=".2f")
    return fig


def acf_pacf_plot(data: dict | None, title: str | None = None) -> go.Figure | None:
    """
    Stacked ACF and PACF stem plots with fixed significance bounds.

    Lags start at 1 (lag 0 equals 1 by definition) and the dashed red lines mark
    the constant white-noise bound +-z/sqrt(n), which does not widen with the lag.

    Args:
        data (dict | None): Output of `econometrics.acf_pacf` (lags/acf/pacf/conf).
        title (str | None): Figure title.

    Returns:
        go.Figure | None: The figure, or None if the input is missing.
    """
    if data is None:
        return None
    lags = np.asarray(data["lags"])
    conf = float(data["conf"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=["ACF", "PACF"])
    for row, key in [(1, "acf"), (2, "pacf")]:
        fig.add_bar(
            x=lags,
            y=np.asarray(data[key]),
            marker_color="#4c78a8",
            width=0.15,
            showlegend=False,
            hovertemplate=f"Lag %{{x}}<br>{key.upper()}: %{{y:.2f}}<extra></extra>",
            row=row,
            col=1,
        )
        for bound in (conf, -conf):
            fig.add_hline(y=bound, line_dash="dash", line_color="#d62728", row=row, col=1)
    fig.update_xaxes(title_text="Lag", dtick=1, row=2, col=1)
    fig.update_layout(template=TEMPLATE, title=title or "ACF / PACF", height=460)
    return fig


def garch_volatility_plot(
    forecast: dict | None,
    forecast_index=None,
    title: str | None = None,
    markers: bool = False,
) -> go.Figure | None:
    """
    In-sample conditional volatility with an optional forecast continuation.

    Args:
        forecast (dict | None): Output of `econometrics.garch_forecast`
            ('fitted_volatility' Series, 'horizon', 'volatility').
        forecast_index: Dates for the forecast horizon; when given, the forecast
            volatility is drawn on the same date axis after the in-sample part.
        title (str | None): Figure title.
        markers (bool): Also draw point markers on the forecast trace;

    Returns:
        go.Figure | None: The figure, or None if the input is missing.
    """
    if forecast is None:
        return None
    fitted = forecast["fitted_volatility"]
    fig = go.Figure()
    if fitted is not None and len(fitted) > 0:
        fig.add_scatter(
            x=fitted.index, y=fitted.to_numpy(), mode="lines", name="In-sample volatility", line={"color": "#1f77b4"}
        )
    x = forecast_index if forecast_index is not None else forecast["horizon"]
    y = np.asarray(forecast["volatility"])
    if fitted is not None and len(fitted) > 0 and forecast_index is not None:
        x = [fitted.index[-1], *list(forecast_index)]
        y = np.concatenate([[float(fitted.iloc[-1])], y])
    fig.add_scatter(
        x=x,
        y=y,
        mode="lines+markers" if markers else "lines",
        name="Forecast volatility",
        line={"color": "#d62728", "dash": "dash"},
    )
    fig.update_layout(template=TEMPLATE, title=title or "Conditional volatility (GARCH)", yaxis_title="Volatility")
    fig.update_traces(yhoverformat=".3f")
    return fig


def shap_local_bar(contributions: pd.Series | None, top_n: int = 12, title: str | None = None) -> go.Figure | None:
    """
    Signed SHAP contributions for a single observation as a horizontal bar.

    Positive contributions (green) push the prediction above the average, negative
    (red) pull it below; the largest-magnitude features are shown.

    Args:
        contributions (pd.Series | None): Signed per-feature SHAP values.
        top_n (int): Number of largest-magnitude features to show.
        title (str | None): Figure title.

    Returns:
        go.Figure | None: The figure, or None if there is nothing to plot.
    """
    if contributions is None or contributions.empty:
        return None
    order = contributions.abs().sort_values(ascending=False).index
    top = contributions.reindex(order).head(top_n).iloc[::-1]
    colors = ["#54a24b" if v >= 0 else "#e45756" for v in top.to_numpy()]
    fig = go.Figure(
        go.Bar(
            x=top.to_numpy(),
            y=top.index.astype(str),
            orientation="h",
            marker={"color": colors},
            hovertemplate="%{y}<br>SHAP: %{x:+.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="rgba(0,0,0,0.4)")
    fig.update_layout(
        template=TEMPLATE,
        title=title or "Local SHAP contributions",
        xaxis_title="Contribution to prediction",
        yaxis_title="",
    )
    return fig


def partial_dependence_plot(
    data: dict | None, labels=None, feature: str | None = None, title: str | None = None
) -> go.Figure | None:
    """
    Partial dependence line(s) for one feature.

    One line for regression; one line per class for multiclass classification.

    Args:
        data (dict | None): Output of `explain.partial_dependence_data` ('grid' and
            'average', one row per output).
        labels (list | None): Class labels for a multi-row average.
        feature (str | None): Feature name for the axis label.
        title (str | None): Figure title.

    Returns:
        go.Figure | None: The figure, or None if there is no data.
    """
    if data is None:
        return None
    grid = np.asarray(data["grid"])
    average = np.asarray(data["average"])
    if average.ndim == 1:
        average = average[None, :]
    fig = go.Figure()
    for i in range(average.shape[0]):
        name = str(labels[i]) if labels is not None and i < len(labels) else "Prediction"
        fig.add_scatter(x=grid, y=average[i], mode="lines", name=name)
    fig.update_layout(
        template=TEMPLATE,
        title=title or f"Partial dependence - {feature}",
        xaxis_title=feature,
        yaxis_title="Average prediction",
        showlegend=average.shape[0] > 1,
    )
    fig.update_traces(xhoverformat=".3f", yhoverformat=".3f")
    return fig
