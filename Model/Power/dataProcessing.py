from __future__ import annotations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import seaborn as sns
from matplotlib import pyplot as plt

from Logger.Logger import mylogger
from Asset_Modeling.Energy_Modeling.data.data import fetchGenerationHistoryData

def plot_hexbin_density(df, x_col, y_col, gridsize=60, cmap='YlGnBu', quantile_clip=None):
    """
    Plots a hexbin density plot for given x and y columns.

    Returns:
        quantile_clip used (float)
    """
    plt.figure(figsize=(10, 6))
    hb = plt.hexbin(df[x_col], df[y_col], gridsize=gridsize, cmap=cmap, mincnt=1)

    counts = hb.get_array()
    if quantile_clip is None:
        quantile_clip = auto_quantile_clip(counts)
    vmax = np.quantile(counts, quantile_clip)
    hb.set_clim(0, vmax)

    cb = plt.colorbar(hb)
    cb.set_label(f'Density (clipped at {quantile_clip:.2f})')

    return quantile_clip

def plot_outliers(df_outliers, x_col, y_col):
    """
    Overlays outlier points on the existing plot.
    """
    if not df_outliers.empty:
        plt.scatter(df_outliers[x_col], df_outliers[y_col], color='red', s=10, alpha=0.6, label='Outliers')
        plt.legend()

def dataRESGenerationCleaning(df, x_col, y_col, gridsize=60, cmap='YlGnBu', quantile_clip=None, bins=1000):
    """
    Detects outliers and plots a density + outlier overlay plot.

    Returns:
        pd.DataFrame: Outliers detected
    """
    if y_col not in ['WIND', 'SR']:
        mylogger.logger.error('y_col has to be either SR or WIND.')
        return pd.DataFrame()

    # Select the appropriate outlier detection method
    if y_col == 'SR':
        outlier_df = outliersDetectionSR(df, x_col, y_col, bins=bins)
    else:  # y_col == 'WIND'
        outlier_df = outliersDetectionWIND(df, x_col, y_col)

    # Plot hexbin and get used clip
    used_clip = plot_hexbin_density(df, x_col, y_col, gridsize, cmap, quantile_clip)

    # Plot outliers
    plot_outliers(outlier_df, x_col, y_col)

    # Final plot formatting
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Density Plot with Outliers: {y_col} vs {x_col}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    mylogger.logger.info(f"Used quantile_clip: {used_clip:.2f}")
    return outlier_df



def auto_quantile_clip(density_values, min_clip=0.5, max_clip=0.99):
    """
    Automatically determine the best quantile_clip value for color scaling in density plots.

    Why:
    - Hexbin plots are often dominated by a few high-density bins (e.g., all the data near (0, 0)).
    - Using the full range for color scaling hides important lower-density structure.
    - This function computes a good clip point based on spread and skew of density values.

    Parameters:
        density_values (array-like): The raw density values from a hexbin plot (e.g., `hb.get_array()`)
        min_clip (float): Lower bound of clip range (default = 0.5)
        max_clip (float): Upper bound of clip range (default = 0.99)

    Returns:
        float: Chosen quantile_clip value (between min_clip and max_clip)
    """
    q50 = np.quantile(density_values, 0.50)
    q95 = np.quantile(density_values, 0.95)
    std = np.std(density_values)
    iqr = q95 - q50

    # Heuristics: more skewed distribution gets lower clip
    if std / (iqr + 1e-6) > 4:
        clip = 0.92
    elif std / (iqr + 1e-6) > 2:
        clip = 0.96
    else:
        clip = 0.98

    return min(max(clip, min_clip), max_clip)


def detect_outliers_iqr(df, x_col, y_col, bins=20):
    """
    Detect outliers in y_col using the IQR method, within bins of x_col.

    Why:
    - The distribution of y_col may vary significantly with x_col.
    - Applying the IQR method per x_col bin accounts for conditional structure (e.g., SR depends on Solar_Radiation).
    - This avoids labeling all low/high values as outliers when they are normal at specific x levels.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        x_col (str): Column to bin (e.g., 'Solar_Radiation')
        y_col (str): Column for which to detect outliers (e.g., 'SR')
        bins (int): Number of bins to divide x_col

    Returns:
        pd.DataFrame: Outlier rows
    """
    df = df.copy()
    df['x_bin'] = pd.cut(df[x_col], bins=bins)

    outliers = []
    for _, group in df.groupby('x_bin'):
        q1 = group[y_col].quantile(0.25)
        q3 = group[y_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers.append(group[(group[y_col] < lower) | (group[y_col] > upper)])

    return pd.concat(outliers) if outliers else pd.DataFrame()

def outliersDetectionWIND(
    df, x_col='Wind_Speed', y_col='Wind_Generation',
    degree=3, z_thresh=3.8,
    low_speed_thresh=10.0, iqr_bins=10
):

    df = df.copy().dropna(subset=[x_col, y_col])

    X = df[[x_col]].values
    y = df[y_col].values
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)

    residuals = y - y_pred
    df['residual'] = residuals

    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    mad_score = np.abs(residuals - median) / mad if mad != 0 else np.zeros_like(residuals)

    df['mad_score'] = mad_score
    model_outliers = df[df['mad_score'] > z_thresh]

    lowwind_df = df[df[x_col] < low_speed_thresh].copy()
    lowwind_df['bin'] = pd.cut(lowwind_df[x_col], bins=iqr_bins)

    iqr_outliers = []
    for bin_interval, group in lowwind_df.groupby('bin'):
        if group.empty or pd.isna(bin_interval):
            continue
        bin_mid = bin_interval.mid
        if bin_mid < 7.0:
            iqr_outliers.append(group[group[y_col] > 0.0])
        else:
            q1 = group[y_col].quantile(0.25)
            q3 = group[y_col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            iqr_outliers.append(group[(group[y_col] < lower) | (group[y_col] > upper)])

    iqr_outliers_df = pd.concat(iqr_outliers) if iqr_outliers else pd.DataFrame()

    combined_outliers = pd.concat([model_outliers, iqr_outliers_df]).drop_duplicates()

    return combined_outliers


def detect_outliers_mad(df, x_col, y_col, bins=100, z_thresh=2.5):
    """
    Detect outliers using robust Z-score based on MAD (median absolute deviation)
    within bins of x_col.

    Parameters:
        df (pd.DataFrame): Data
        x_col (str): Column to bin
        y_col (str): Column to check for outliers
        bins (int): Number of bins to divide x_col
        z_thresh (float): Robust Z-score threshold

    Returns:
        pd.DataFrame: Outlier rows
    """
    df = df.copy()
    df['x_bin'] = pd.cut(df[x_col], bins=bins)
    outliers = []

    for _, group in df.groupby('x_bin'):
        median = group[y_col].median()
        mad = np.median(np.abs(group[y_col] - median))
        if mad == 0:
            continue
        # robust_z = 0.6745 * (group[y_col] - median) / mad
        # outliers.append(group[np.abs(robust_z) > z_thresh])
        mad_score = np.abs(group[y_col] - median) / mad
        outliers.append(group[mad_score > z_thresh])

    return pd.concat(outliers) if outliers else pd.DataFrame()

def outliersDetectionSR(df, x_col, y_col, bins=20, iqr_threshold=10, z_thresh=2.5):
    """
    Detect outliers in y_col using a hybrid method:
    - IQR for low x_col bins (e.g., low solar radiation)
    - MAD for other bins

    Parameters:
        df (pd.DataFrame): Input DataFrame
        x_col (str): Column to bin (e.g., 'Solar_Radiation')
        y_col (str): Column to check for outliers (e.g., 'SR')
        bins (int): Number of bins to divide x_col
        iqr_threshold (float): x_col threshold below which IQR is used
        z_thresh (float): MAD Z-score threshold

    Returns:
        pd.DataFrame: Outlier rows
    """

    df = df.copy()
    df['x_bin'] = pd.cut(df[x_col], bins=bins)
    outliers = []

    for bin_interval, group in df.groupby('x_bin'):
        if group.empty:
            continue

        # Use the bin midpoint to determine method
        bin_mid = bin_interval.mid
        if bin_mid < 5.0:
            outliers.append(group[group[y_col] > 0.0])
        elif bin_mid < iqr_threshold:
            # IQR method
            q1 = group[y_col].quantile(0.25)
            q3 = group[y_col].quantile(0.50)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers.append(group[(group[y_col] < lower) | (group[y_col] > upper)])
        else:
            # MAD method
            median = group[y_col].median()
            mad = np.median(np.abs(group[y_col] - median))
            if mad == 0:
                continue
            mad_score = np.abs(group[y_col] - median) / mad
            outliers.append(group[mad_score > z_thresh])

    return pd.concat(outliers) if outliers else pd.DataFrame()


def visualize_correlations(df, top_n=10, corr_threshold=0.1, redundancy_threshold=0.9):

    # Keep only numeric columns
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()

    def select_top_features(target):
        # Correlation with target
        target_corr = corr[target].drop(['SR', 'WIND'], errors='ignore')
        target_corr = target_corr[abs(target_corr) > corr_threshold].sort_values(key=abs, ascending=False)

        final_features = []
        removed_redundant = []

        for feature in target_corr.index:
            is_redundant = False
            for kept in final_features:
                if abs(corr.loc[feature, kept]) > redundancy_threshold:
                    is_redundant = True
                    removed_redundant.append((feature, kept, corr.loc[feature, kept]))
                    break
            if not is_redundant:
                final_features.append(feature)
            if len(final_features) == top_n:
                break

        return final_features, target_corr.loc[final_features], removed_redundant

    # Run for both targets
    sr_features, sr_corrs, sr_removed = select_top_features('SR')
    wind_features, wind_corrs, wind_removed = select_top_features('WIND')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x=sr_corrs.abs().values, y=sr_corrs.index, ax=axes[0], hue=sr_corrs.index, legend=False, palette='YlOrBr')
    axes[0].set_title('Top Non-Redundant Features for SR')
    axes[0].set_xlabel('Abs(Pearson Correlation)')

    sns.barplot(x=wind_corrs.abs().values, y=wind_corrs.index, ax=axes[1], hue=wind_corrs.index, legend=False, palette='Blues')
    axes[1].set_title('Top Non-Redundant Features for WIND')
    axes[1].set_xlabel('Abs(Pearson Correlation)')

    plt.tight_layout()
    plt.show()

    print(f"\n✅ Selected top {top_n} features for SR (solar):")
    for f in sr_features:
        print(f"  - {f}")

    print(f"\n✅ Selected top {top_n} features for WIND (wind):")
    for f in wind_features:
        print(f"  - {f}")

    if sr_removed or wind_removed:
        print(f"\n⚠️ Features removed due to redundancy (> {redundancy_threshold} correlation):")
        for feat, kept, val in sr_removed:
            print(f"  SR: {feat} removed (too correlated with {kept}, corr = {val:.2f})")
        for feat, kept, val in wind_removed:
            print(f"  WIND: {feat} removed (too correlated with {kept}, corr = {val:.2f})")
    else:
        print("\n✅ No redundant features removed.")

    return sr_features, wind_features


def plot_box_and_return_outliers(df, group_by_col, value_col, bins=10):
    # Bin if necessary
    if pd.api.types.is_numeric_dtype(df[group_by_col]) and df[group_by_col].nunique() > 20:
        bin_col = f"{group_by_col}_bin"
        df[bin_col] = pd.cut(df[group_by_col], bins=bins)
        group_col = bin_col
    else:
        group_col = group_by_col

    # Create boxplot
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=group_col, y=value_col, data=df, showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
                boxprops=dict(alpha=0.6),
                flierprops=dict(marker='o', color='red', alpha=0.4))
    plt.title(f'Boxplot of {value_col} by {group_by_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # Compute outliers
    outliers = []
    grouped = df.groupby(group_col)
    for _, group in grouped:
        q1 = group[value_col].quantile(0.25)
        q3 = group[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers.append(group[(group[value_col] < lower) | (group[value_col] > upper)])

    outlier_df = pd.concat(outliers) if outliers else pd.DataFrame()
    return outlier_df


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define feature sets (customize as needed)
wind_features = [
    'Wind_Speed_100m', 'Wind_Direction_100m', 'Wind_Gusts_10m',
    'Surface_Pressure', 'Cloud_Cover', 'WIND_capa', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos'
]

sr_features = [
    'Solar_Radiation', 'Direct_Radiation', 'Diffuse_Radiation',
    'Direct_Normal_Irradiance', 'Global_Tilted_Irradiance',
    'Cloud_Cover', 'Temperature_2m', 'SR_capa', 'month_sin', 'month_cos', 'hour_sin', 'hour_cos'
]


def run_pca(feature_list, label):
    X = history[feature_list]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=None)
    X_pca = pca.fit_transform(X_scaled)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'{label} Forecasting - Explained Variance')
    plt.grid(True)
    plt.show()

    # Output loadings
    loadings = pd.DataFrame(pca.components_.T, index=feature_list,
                            columns=[f'PC{i + 1}' for i in range(X_pca.shape[1])])
    print(f"\nTop contributing features to PC1 for {label}:")
    print(loadings['PC1'].sort_values(key=abs, ascending=False).head(5))


if __name__ == '__main__':
    history = fetchGenerationHistoryData('FR')
    # sr_features, wind_features = visualize_correlations(history, top_n=15)
    outlier_indices = set()

    outliers = dataRESGenerationCleaning(history, 'Solar_Radiation', 'SR', quantile_clip=0.9)
    outlier_indices.update(outliers.index.tolist())

    outliers = dataRESGenerationCleaning(history, 'Wind_Speed_100m', 'WIND', quantile_clip=0.9)
    outlier_indices.update(outliers.index.tolist())


    history_cleaned = history.drop(index=outlier_indices)
    plot_hexbin_density(history_cleaned, 'Solar_Radiation', 'SR')
    plt.show()
    plot_hexbin_density(history_cleaned, 'Wind_Speed_100m', 'WIND')
    plt.show()
