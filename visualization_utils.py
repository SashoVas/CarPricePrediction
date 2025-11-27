import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


def display_corr_heatmap(corr_df, figsize=(10, 8), cmap='coolwarm', annot=True, fmt='.2f', vmin=None, vmax=None):
    """Display a heatmap for a square correlation DataFrame using matplotlib."""

    # Basic validation
    if not hasattr(corr_df, 'shape') or corr_df.shape[0] != corr_df.shape[1]:
        raise ValueError(
            'corr_df must be a square DataFrame (correlation matrix)')

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_df.values, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect='auto')

    labels = list(corr_df.columns)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # draw white grid lines between cells
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    # annotations
    if annot:
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = corr_df.iat[i, j]
                txt = format(val, fmt)
                ax.text(j, i, txt, ha='center', va='center',
                        color='black', fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.show()


def display_pie_chart(data, column=None, figsize=(6, 6), colors=None, autopct='%1.1f%%', explode=None, startangle=90, shadow=False, title=None, legend=False, pctdistance=0.6, normalize=False, other_threshold=None):
    """Display a pie chart for counts in a pandas Series or a single-column DataFrame.

    Parameters:
        data: pandas Series or DataFrame with one column
        column: optional column name if a multi-column DataFrame is provided (not required for Series or single-column DataFrame)
        figsize: figure size tuple
        colors: sequence of colors for wedges
        autopct: string or function to label wedges with numeric value
        explode: sequence of offsets for each wedge (same length as categories) or None
        startangle: start angle for first wedge
        shadow: draw a shadow
        title: optional title
        legend: if True, place labels in a legend instead of on wedges
        pctdistance: ratio to place the autopct text
        normalize: if True, use proportions (sum to 1) instead of raw counts
        other_threshold: float between 0 and 1 - categories with fraction < threshold are grouped into 'Other'
    """

    # Accept Series directly
    if isinstance(data, pd.Series):
        series = data.dropna()
        col_name = series.name if series.name is not None else 'value'
    elif isinstance(data, pd.DataFrame):
        # If user passed a DataFrame and specified a column, use it
        if column is not None:
            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            series = data[column].dropna()
            col_name = column
        else:
            # If no column provided, accept only single-column DataFrame
            if data.shape[1] == 1:
                series = data.iloc[:, 0].dropna()
                col_name = data.columns[0]
            else:
                raise ValueError(
                    'Provide a column name or pass a Series or a DataFrame with exactly one column')
    else:
        raise TypeError('data must be a pandas Series or DataFrame')

    if series.empty:
        raise ValueError('No data to plot after dropping NA values')

    counts = series.value_counts()

    # Optionally group small categories into 'Other'
    if other_threshold is not None:
        if not (0 < other_threshold < 1):
            raise ValueError('other_threshold must be between 0 and 1')
        total = counts.sum()
        small_mask = (counts / total) < other_threshold
        if small_mask.any():
            other_sum = counts[small_mask].sum()
            counts = counts[~small_mask]
            counts['Other'] = other_sum

    labels = counts.index.tolist()
    sizes = counts.values.astype(float)

    if normalize:
        sizes = sizes / sizes.sum()

    fig, ax = plt.subplots(figsize=figsize)

    # By default show labels on wedges (so the category values appear on the chart).
    # If legend=True, keep labels out of wedges and place them in a legend instead.
    wedge_labels = None if legend else labels
    pie_result = ax.pie(sizes, labels=wedge_labels, autopct=(autopct if not legend else None),
                        colors=colors, explode=explode, startangle=startangle, shadow=shadow, pctdistance=pctdistance)
    # matplotlib.pie returns (wedges, texts, autotexts) when autopct is set,
    # otherwise it may return only (wedges, texts). Handle both cases.
    if len(pie_result) == 3:
        wedges, texts, autotexts = pie_result
    else:
        wedges, texts = pie_result
        autotexts = []

    if title:
        ax.set_title(title)

    if legend:
        ax.legend(wedges, labels, title=column, loc='best',
                  bbox_to_anchor=(1, 0.5), fontsize=8)

    plt.tight_layout()
    plt.show()


def display_points(df, x_col=None, y_col=None, figsize=(8, 6), color='C0', marker='o', s=30, alpha=0.8, xlabel=None, ylabel=None, title=None, annotations=False, annotate_fmt='.2f'):
    """Plot every row in `df` as a point (x,y) using matplotlib."""

    # Determine columns to use
    if x_col is None or y_col is None:
        if df.shape[1] == 2:
            x_col, y_col = df.columns[0], df.columns[1]
        else:
            raise ValueError(
                'Provide x_col and y_col or supply a DataFrame with exactly two columns.')

    x = df[x_col]
    y = df[y_col]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, c=color, marker=marker, s=s, alpha=alpha)

    ax.set_xlabel(xlabel if xlabel is not None else x_col)
    ax.set_ylabel(ylabel if ylabel is not None else y_col)
    if title:
        ax.set_title(title)

    if annotations:
        for xx, yy in zip(x, y):
            try:
                label = f'{format(xx, annotate_fmt)}, {format(yy, annotate_fmt)}'
            except Exception:
                label = f'{xx}, {yy}'
            ax.annotate(label, (xx, yy), textcoords="offset points",
                        xytext=(3, 3), fontsize=8)

    plt.tight_layout()
    plt.show()


def display_boxplots(df, columns=None, figsize=(12, 6), layout=None, showfliers=True, vert=True, notch=False, grid=True, title=None):
    """Create a box plot for each column in `columns` (or all numeric columns if None).
    Parameters:
        df: pandas DataFrame
        columns: list of column names to plot (default: numeric columns)
        figsize: tuple for the figure size (width, height)
        layout: (nrows, ncols) to arrange subplots; auto-chosen if None
        showfliers, vert, notch, grid: passed to matplotlib.boxplot
        title: optional overall title for the figure"""

    # Select columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # ensure provided columns exist
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f'Columns not found in DataFrame: {missing}')

    if len(columns) == 0:
        raise ValueError('No columns to plot')

    n = len(columns)
    if layout is None:
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)
    else:
        nrows, ncols = layout
        if nrows * ncols < n:
            raise ValueError(
                'Provided layout is too small for number of columns')

    # Create subplots; scale height by number of rows
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        figsize[0], figsize[1] * nrows / max(1, ncols)))
    # flatten axes array for easy indexing
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for i, col in enumerate(columns):
        ax = axes_flat[i]
        data = df[col].dropna()
        if data.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(col)
            ax.set_xticks([])
            continue
        ax.boxplot(data, vert=vert, showfliers=showfliers, notch=notch)
        ax.set_title(col)
        if grid:
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # if vertical, hide x tick labels except maybe the column name
        if vert:
            ax.set_xticks([])

    # turn off any unused axes
    for j in range(n, nrows * ncols):
        ax = axes_flat[j]
        ax.set_visible(False)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    # if suptitle used, adjust layout
    if title:
        plt.subplots_adjust(top=0.92)
    plt.show()


def display_histogram(df, column=None, bins=30, figsize=(8, 6), color='C0', alpha=0.8, density=False, cumulative=False, log=False, xlabel=None, ylabel='Count', title=None, show_stats=True):
    """Display a histogram for a single column DataFrame or a specified column.

    Parameters:
        df: pandas DataFrame containing data
        column: column name to plot. If None, df must have exactly one column.
        bins: number of histogram bins or sequence of bin edges
        figsize: tuple for figure size
        color: bar color
        alpha: bar alpha transparency
        density: if True, plot probability density instead of counts
        cumulative: if True, plot the cumulative histogram
        log: if True, use a log scale for the y axis
        xlabel, ylabel, title: axis labels and title
        show_stats: if True, draw vertical lines for mean and median and annotate them
    """

    # Validation and column selection
    if column is None:
        if df.shape[1] == 1:
            column = df.columns[0]
        else:
            raise ValueError(
                'Provide a column name or pass a DataFrame with exactly one column')
    else:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

    data = df[column].dropna()
    if data.empty:
        raise ValueError('No data to plot after dropping NA values')

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(data, bins=bins, color=color, alpha=alpha,
            density=density, cumulative=cumulative)

    if log:
        ax.set_yscale('log')

    if show_stats and not cumulative:
        try:
            mean = data.mean()
            median = data.median()
            ax.axvline(mean, color='C1', linestyle='--',
                       linewidth=1, label=f'Mean: {mean:.2f}')
            ax.axvline(median, color='C2', linestyle=':',
                       linewidth=1, label=f'Median: {median:.2f}')
            # place legend for stats
            ax.legend(fontsize=8)
        except Exception:
            # if numeric computations fail, skip stats
            pass

    ax.set_xlabel(xlabel if xlabel is not None else column)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()
