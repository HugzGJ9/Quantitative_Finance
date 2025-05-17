import random

import pandas as pd
from matplotlib import pyplot as plt
from Model.Power.dataProcessing import plot_hexbin_density
from API.SUPABASE.client import getDfSupabase

import numpy as np
import pandas as pd
import scipy.stats as st

from Logger.Logger import mylogger

weather_data = getDfSupabase('WeatherFR')
weather_data['id'] = pd.to_datetime(weather_data['id'], utc=True)
weather_data = weather_data.set_index('id')
weather_data.index.name = 'time'
weather_data = weather_data[['Solar_Radiation', 'Cloud_Cover']]

autumn = weather_data[(weather_data.index.month<12) & (weather_data.index.month>8)]
winter = weather_data[(weather_data.index.month> 11) | (weather_data.index.month<3)]
summer = weather_data[(weather_data.index.month>5) & (weather_data.index.month<9)]
spring = weather_data[(weather_data.index.month>2) & (weather_data.index.month<6)]
def best_fit_distribution(series, *, candidate_cont=None, candidate_disc=None,
                           ks_alpha=0.05, bins=50, random_state=None):
    """
    Identify the best-fitting distribution for a 1-D sample.
    Returns a dict with keys: family, name, params, statistic, p_value, test.
    """
    rng = np.random.default_rng(random_state)
    x = pd.Series(series).dropna().to_numpy()
    def _safe_fit(dist):
        """Use dist.fit() when it exists, else manual MLE for Poisson/Geom/NB."""
        if callable(getattr(dist, "fit", None)):
            try:
                return dist.fit(x, random_state=rng)
            except TypeError:
                return dist.fit(x)

        name, μ, σ2 = dist.name, np.mean(x), np.var(x, ddof=1)
        if name == "poisson":
            return (μ,)
        if name == "geom":
            return (1.0/μ,)
        if name == "nbinom":
            if σ2 <= μ:
                raise RuntimeError("over-dispersion required for NB fit")
            p = μ/σ2
            n = μ * p / (1.0 - p)
            return (n, p)
        raise AttributeError(f"{name}.fit() missing and no manual recipe")

    if np.allclose(x, x[0], atol=1e-12):
        mylogger.logger.info("Sample is constant → degenerate distribution")
        return dict(family='degenerate', name='degenerate',
                    params=(x[0],), statistic=0.0, p_value=1.0, test='constant')

    looks_int  = np.allclose(x, np.round(x))
    non_neg    = (x >= 0).all()
    use_disc   = looks_int and non_neg

    if candidate_cont is None:
        candidate_cont = ['norm', 'lognorm', 'gamma', 'beta',
                          'weibull_min', 'weibull_max', 'expon', 'cauchy', 't']
    if candidate_disc is None:
        candidate_disc = ['poisson', 'geom', 'nbinom', ]

    best, best_stat = None, np.inf
    mylogger.logger.info(f"Fitting {'discrete' if use_disc else 'continuous'} families")

    if use_disc:
        for name in candidate_cont:
            dist = getattr(st, name)
            try:
                params = _safe_fit(dist)
                D, p = st.kstest(x, name, args=params, N=len(x))
                mylogger.logger.debug(f"{name:12s}  KS D={D:.4g}")
                if D < best_stat:
                    best = dict(family=dist, name=name, params=params,
                                statistic=D, p_value=p, test='ks')
                    best_stat = D
            except Exception as err:
                mylogger.logger.debug(f"{name:12s}  skipped ({err})")

    else:
        for name in candidate_disc:
            dist = getattr(st, name)
            try:
                params = _safe_fit(dist)
                ll = np.mean(dist.logpmf(x, *params))
                mylogger.logger.debug(f"{name:12s}  mean log-lik={ll:.4g}")
                if -ll < best_stat:                             # maximise ll
                    best = dict(family=dist, name=name, params=params,
                                statistic=ll, p_value=None, test='ll')
                    best_stat = -ll
            except Exception as err:
                mylogger.logger.debug(f"{name:12s}  skipped ({err})")

    if best is None:
        raise RuntimeError("No distribution could be fitted.")

    mylogger.logger.info(f"Best fit → {best['name']}  params={best['params']}")
    return best

def plot_distribution_fit(series, fit_result, *, bins=50, ax=None,
                          hist_kws=None, stem_kwargs=None):
    """
    Draw a normalised histogram of `series` and overlay the fitted distribution.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    data = pd.Series(series).dropna().to_numpy()
    hist_kws = hist_kws or {}
    stem_kwargs = stem_kwargs or {}

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(data, bins=bins, density=True, alpha=0.6, **hist_kws)

    dist, params = fit_result['family'], fit_result['params']
    xmin, xmax = data.min(), data.max()

    if fit_result['test'] == 'ks':
        xs = np.linspace(xmin, xmax, 1_000)
        ax.plot(xs, dist.pdf(xs, *params), linewidth=2)
    else:
        xs = np.arange(int(xmin), int(xmax) + 1)
        markerline, stemlines, _ = ax.stem(xs,
                                           dist.pmf(xs, *params),
                                           basefmt=" ", use_line_collection=True,
                                           **stem_kwargs)
        plt.setp(stemlines, linewidth=1.5)

    ax.set_title(f"Data vs. {fit_result['name']} fit")
    ax.set_ylabel("Density")
    ax.set_xlabel("Value")
    ax.margins(x=0.01)
    return ax

def pltSolarGen():
    rad = np.linspace(0, 900, 300)
    lf = simulate_solar_generation(rad, 1.0)

    plt.plot(rad, lf)
    plt.xlabel("Solar Radiation (W/m²)")
    plt.ylabel("Load Factor")
    plt.title("Solar Generation Curve")
    plt.grid(True)
    plt.show()

def simulate_solar_generation(radiation, capacity, threshold=50, radiation_max=850, coef=1):

    is_scalar = np.isscalar(radiation)
    radiation = np.asarray(radiation)

    norm = (radiation - threshold) / (radiation_max - threshold) * coef
    norm = np.clip(norm, 0, 1)

    load_factor = norm * capacity

    return float(load_factor) if is_scalar else load_factor
def simulate_solar_generation_convex(radiation, capacity, threshold=50, radiation_max=850, scale=0.005):
    is_scalar = np.isscalar(radiation)
    radiation = np.asarray(radiation)

    # Base load factor from exponential
    raw_factor = 1 - np.exp(-scale * (radiation - threshold))
    raw_factor[radiation < threshold] = 0.0
    raw_factor[radiation > radiation_max] = 1.0

    # Final output clipped between 0 and 1, then scaled by capacity
    load_factor = np.clip(raw_factor, 0, 1)
    power_output = load_factor * capacity

    return float(power_output) if is_scalar else power_output

def buildSyntheticGeneration_save(df: pd.DataFrame,
                             avg_target: float,
                             capacity: float = 10.0,
                             sat_threshold: float = 850.0,
                             noise_sigma: float = 0.10) -> pd.DataFrame:
    out = df.copy()
    out['generation'] = 0.0

    if avg_target == 0.0 or 'Solar_Radiation' not in df.columns:
        return out
    valid = out.dropna(subset=['Solar_Radiation'])

    if valid.empty:
        return out

    radiation = np.minimum(valid['Solar_Radiation'].to_numpy(), sat_threshold)
    generation = (radiation / sat_threshold) * capacity
    generation[valid['Solar_Radiation'].to_numpy() < 50.0] = 0.0
    if np.all(generation == 0.0):
        return out
    noise = np.random.normal(loc=1.0, scale=noise_sigma, size=generation.shape)
    generation *= noise

    # Calcul des bornes physiques
    lower_bound, upper_bound = simulate_solar_generation(
        np.array([valid['Solar_Radiation'].min(), valid['Solar_Radiation'].max()]),
        capacity=capacity,
        threshold=50,
        radiation_max=850
    )

    current_mean = np.mean(generation)
    max_mask = generation >= capacity * 0.98  # tolérance de 2% pour saturation
    non_max_mask = ~max_mask
    non_max_mean = np.mean(generation[non_max_mask]) if np.any(non_max_mask) else 0.0

    if non_max_mean > 0:
        generation[non_max_mask] *= avg_target / current_mean

    generation = np.clip(generation, a_min=lower_bound, a_max=upper_bound)
    out.loc[valid.index, 'generation'] = generation
    return out

def buildSyntheticGeneration(df: pd.DataFrame,
                             avg_target: float,
                             capacity: float = 10.0,
                             sat_threshold: float = 850.0,
                             noise_sigma: float = 0.01) -> pd.DataFrame:
    out = df.copy()
    out['generation'] = 0.0

    if avg_target == 0.0 or 'Solar_Radiation' not in df.columns:
        return out
    valid = out.dropna(subset=['Solar_Radiation'])

    if valid.empty:
        return out

    generation = simulate_solar_generation(valid['Solar_Radiation'],
                                           capacity=capacity,
                                           threshold=50,
                                           radiation_max=850
                                           )
    if np.all(generation == 0.0):
        return out

    lower_bound, upper_bound = simulate_solar_generation(
        np.array([valid['Solar_Radiation'].min(), valid['Solar_Radiation'].max()]),
        capacity=capacity,
        threshold=50,
        radiation_max=850
    )
    max_mask = generation >= capacity * 0.98  # tolérance de 2% pour saturation
    non_max_mask = ~max_mask
    non_max_mean = np.mean(generation[non_max_mask]) if np.any(non_max_mask) else 0.0

    if non_max_mean > 0:
        generation[non_max_mask] *= avg_target / non_max_mean

    generation = np.clip(generation, a_min=lower_bound, a_max=upper_bound)
    out.loc[valid.index, 'generation'] = generation

    return out
#
# df = winter[winter.index.hour == 12]
# result = best_fit_distribution(df['Solar_Radiation'])   # or save.squeeze() if it's a Series
# plot_distribution_fit(df['Solar_Radiation'], result, bins=50)
# plt.show()
# mean_target = 2.6
# save = buildSyntheticGeneration(df, mean_target, capacity=10.0, sat_threshold=950)
# plt.show()
# plot_hexbin_density(save, 'Solar_Radiation', 'generation')
# plt.show()
