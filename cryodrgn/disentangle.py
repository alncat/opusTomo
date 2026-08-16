'''
Metrics for the relationship between the composition and conformation latent codes.

Two questions are asked of a trained model:

  1. How much of either latent code is recoverable from the other?  Regressors are fitted in
     both directions and scored on held-out particles, against a permutation null.
  2. Does clustering in one code carry information about the other?  The two codes are
     clustered independently and the composition-class mixture within each conformation class
     is compared with the global mixture, again against a permutation null.

Effect sizes are reported against a null rather than p-values: at N~1e5 any non-zero coupling
is significant, so the question is how large it is, not whether it exists.

The functions here take plain arrays and have no CLI or file-format dependencies, so they can
be exercised directly on synthetic codes.
'''

import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, r2_score
from sklearn.model_selection import train_test_split

# Regression is the expensive step and converges long before the full particle set is used.
MAX_REGRESSION_PARTICLES = 30000
DEFAULT_PERMUTATIONS = 200


def standardize(x):
    '''Zero mean, unit variance per dimension.  Guards against constant dimensions.'''
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


def predictive_r2(source, target, model, seed=0, test_size=0.3):
    '''Fraction of the target's variance recoverable from the source on held-out particles.

    Variance is pooled across the target's dimensions rather than averaged per-dimension, so a
    dimension carrying little variance cannot dominate the score.'''
    a_tr, a_te, b_tr, b_te = train_test_split(source, target, test_size=test_size,
                                              random_state=seed)
    model.fit(a_tr, b_tr)
    return float(r2_score(b_te, model.predict(a_te), multioutput='variance_weighted'))


def _ridge():
    return Ridge(alpha=1.0)


def _forest(seed=0):
    return RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed)


def recoverability(comp, conf, seed=0, max_particles=MAX_REGRESSION_PARTICLES):
    '''Predictive R^2 in both directions, with a linear and a nonlinear regressor.

    Ridge and random forest agreeing means any coupling is essentially linear; the random
    forest scoring much higher would indicate nonlinear structure the linear fit misses.'''
    rng = np.random.default_rng(seed)
    n = len(comp)
    if n > max_particles:
        idx = rng.choice(n, max_particles, replace=False)
        comp, conf = comp[idx], conf[idx]

    out = {
        'ridge_comp_to_conf': predictive_r2(comp, conf, _ridge(), seed),
        'forest_comp_to_conf': predictive_r2(comp, conf, _forest(seed), seed),
        'ridge_conf_to_comp': predictive_r2(conf, comp, _ridge(), seed),
        'forest_conf_to_comp': predictive_r2(conf, comp, _forest(seed), seed),
    }
    # Null: the same regressor on codes paired at random.  Slightly negative, not zero,
    # because a fitted model scored on unrelated targets does worse than predicting the mean.
    shuffled = conf[rng.permutation(len(conf))]
    out['null'] = predictive_r2(comp, shuffled, _forest(seed), seed)
    out['n_particles'] = int(len(comp))
    return out


def cluster(x, k, seed=0):
    return KMeans(k, n_init=10, random_state=seed).fit_predict(x)


def class_coupling(labels_comp, labels_conf, kc, kf, seed=0,
                   n_permutations=DEFAULT_PERMUTATIONS):
    '''Composition-class mixture within each conformation class, against a permutation null.

    Total-variation distance from the global mixture is 0 when a conformation class draws its
    particles in the same proportions as the dataset as a whole, and approaches 1 when it is
    dominated by composition classes that are globally rare.'''
    rng = np.random.default_rng(seed)
    n = len(labels_comp)
    global_mix = np.bincount(labels_comp, minlength=kc) / n

    mixtures, sizes = [], []
    for b in range(kf):
        sel = labels_conf == b
        sizes.append(int(sel.sum()))
        if not sel.any():
            mixtures.append(np.full(kc, np.nan))
            continue
        mixtures.append(np.bincount(labels_comp[sel], minlength=kc) / sel.sum())
    mixtures = np.asarray(mixtures)
    tv = 0.5 * np.abs(mixtures - global_mix).sum(1)

    # Null: the same statistic with the conformation labels shuffled, which destroys any
    # association while preserving both class-size distributions.
    null = []
    for _ in range(n_permutations):
        permuted = labels_conf[rng.permutation(n)]
        null.append(np.mean([
            0.5 * np.abs(np.bincount(labels_comp[permuted == b], minlength=kc)
                         / max((permuted == b).sum(), 1) - global_mix).sum()
            for b in range(kf)]))

    return {
        'global_mixture': global_mix,
        'mixtures': mixtures,
        'class_sizes': sizes,
        'tv': tv,
        'tv_null_mean': float(np.mean(null)),
        'tv_null_p95': float(np.percentile(null, 95)),
        'ami': float(adjusted_mutual_info_score(labels_comp, labels_conf)),
        'ari': float(adjusted_rand_score(labels_comp, labels_conf)),
    }


def rank_conformation_classes(coupling):
    '''Conformation classes ordered by how far their composition mixture departs from global.

    Returns (class index, TV, enrichment over null, size, dominant composition class, its
    fraction, its global fraction), most coupled first.  This is the table a user reads to
    decide which class to pass to --class.'''
    tv, mixtures = coupling['tv'], coupling['mixtures']
    global_mix, sizes = coupling['global_mixture'], coupling['class_sizes']
    null = coupling['tv_null_mean']

    rows = []
    for b in np.argsort(-tv):
        mix = mixtures[b]
        if np.all(np.isnan(mix)):
            continue
        dom = int(np.argmax(mix))
        rows.append({
            'conformation_class': int(b),
            'tv': float(tv[b]),
            'tv_over_null': float(tv[b] / null) if null > 0 else float('inf'),
            'size': sizes[b],
            'dominant_composition_class': dom,
            'dominant_fraction': float(mix[dom]),
            'dominant_fraction_global': float(global_mix[dom]),
            'enrichment': float(mix[dom] / global_mix[dom]) if global_mix[dom] > 0 else float('inf'),
        })
    return rows


def run_diagnostics(comp, conf, kc=10, kf=8, seed=0, n_permutations=DEFAULT_PERMUTATIONS,
                    max_particles=MAX_REGRESSION_PARTICLES):
    '''Full stage-1 analysis.  comp and conf are the raw (N, d) latent codes.'''
    comp_s, conf_s = standardize(comp), standardize(conf)
    labels_comp = cluster(comp_s, kc, seed)
    labels_conf = cluster(conf_s, kf, seed)
    coupling = class_coupling(labels_comp, labels_conf, kc, kf, seed, n_permutations)
    return {
        'n_particles': int(len(comp)),
        'dim_composition': int(comp.shape[1]),
        'dim_conformation': int(conf.shape[1]),
        'kc': kc,
        'kf': kf,
        'seed': seed,
        'recoverability': recoverability(comp_s, conf_s, seed, max_particles),
        'coupling': coupling,
        'ranking': rank_conformation_classes(coupling),
        'labels_composition': labels_comp,
        'labels_conformation': labels_conf,
    }
