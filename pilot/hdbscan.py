"""Density-based pilot on unchanged named-party shares.

Candidate selection prioritizes coverage rather than a silhouette calculated on
an arbitrarily small retained subset. One-cluster and all-noise outcomes remain
valid diagnostics. Density membership strengths are not class probabilities.
"""
from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import json
import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from threadpoolctl import threadpool_limits

from .common import RESULTS, Evaluator, clean_json, dataset_keys, finalize, load_dataset, write_json


def configuration_grid(n: int):
    sizes = sorted({max(10, round(n * f)) for f in (.005, .01, .02, .05)})
    for size, samples, selection in product(sizes, (5, 15), ('eom', 'leaf')):
        yield dict(min_cluster_size=size, min_samples=samples,
                   cluster_selection_method=selection, allow_single_cluster=True,
                   metric='euclidean', n_jobs=1, copy=True)


def fit_model(X, params):
    return HDBSCAN(**params).fit(X)


def choose_candidate(candidates, selection_method):
    """Lexicographic coverage protection, then separation within 2 pp coverage.

    A single cluster can win if all actual subdivisions lose >2 pp coverage.
    This does not assert that a low-coverage winner explains the full dataset.
    """
    choices = [r for r in candidates if r['params']['cluster_selection_method'] == selection_method]
    max_coverage = max(r['metrics']['coverage'] for r in choices)
    retained = [r for r in choices if r['metrics']['coverage'] >= max_coverage - .02]
    separable = [r for r in retained if r['metrics']['k'] >= 2 and np.isfinite(r['metrics']['silhouette'])]
    if separable:
        winner = max(separable, key=lambda r: (r['metrics']['silhouette'], r['metrics']['coverage'],
                                               -r['metrics']['k'], r['params']['min_cluster_size']))
    else:
        winner = max(retained, key=lambda r: (r['metrics']['coverage'], -r['metrics']['k'],
                                              r['params']['min_cluster_size']))
    return winner, dict(rule='Within each eom/leaf family retain configurations within 0.02 of maximum assigned coverage; among these prefer highest finite silhouette with at least two clusters, otherwise highest coverage diagnostic outcome. Do not compare this silhouette to full-coverage algorithms without coverage.',
                       maximum_family_coverage=max_coverage,
                       coverage_tolerance=.02,
                       all_noise=winner['metrics']['k'] == 0,
                       single_cluster=winner['metrics']['k'] == 1,
                       low_coverage_warning=winner['metrics']['coverage'] < .8,
                       cluster_size_note='min_cluster_size limits the number of UIC observations in a density cluster; it does not remove small-electorate precincts.',
                       stability_note='Absolute min_cluster_size and min_samples remain fixed across 80% subsamples; no predictive assignment to excluded UICs is invented.')


def run_dataset(key, repeats=30):
    ds = load_dataset(key)
    evaluator = Evaluator(ds)
    base = RESULTS / 'hdbscan' / key
    base.mkdir(parents=True, exist_ok=True)
    candidates, failures = [], []
    start = time.time()
    for num, params in enumerate(configuration_grid(len(ds.X))):
        t0 = time.time()
        try:
            model = fit_model(ds.X, params)
            labels = model.labels_.copy()
            metrics = evaluator.score(labels)
            candidate = dict(candidate=num, params=params, metrics=metrics,
                             fit_seconds=time.time()-t0, labels=labels,
                             strengths=model.probabilities_.copy())
            candidates.append(candidate)
        except Exception as exc:
            failures.append(dict(candidate=num, params=params, error=repr(exc)))
        records = [dict(candidate=r['candidate'], **{'param_'+k:v for k,v in r['params'].items()}, **r['metrics'],
                        fit_seconds=r['fit_seconds']) for r in candidates]
        pd.DataFrame(records).to_csv(base/'candidates.csv', index=False)
    write_json(base/'grid_failures.json', failures)
    if not candidates:
        raise RuntimeError(f'All HDBSCAN candidates failed: {key}')
    print(json.dumps(dict(stage='grid', dataset=key, candidates=len(candidates),
                          failures=len(failures), seconds=time.time()-start)), flush=True)
    output = []
    for family in ('eom', 'leaf'):
        if not any(r['params']['cluster_selection_method'] == family for r in candidates):
            continue
        winner, rule = choose_candidate(candidates, family)
        params = winner['params'].copy()

        def refit(X, seed, params=params):
            return fit_model(X, params).labels_

        info = finalize(ds, 'hdbscan', params, winner['labels'], refit, evaluator,
                        variant=family, extras=dict(selection=rule, candidate=winner['candidate'],
                                                    grid_candidates=len(candidates), grid_failures=failures),
                        repeats=repeats)
        strength = pd.DataFrame({'uik': ds.meta.uik,
                                 'density_membership_strength': winner['strengths']})
        strength.to_csv(base/family/'density_strength.csv', index=False, encoding='utf-8-sig')
        output.append(info)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='append', default=None)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    rows = []
    with threadpool_limits(limits=1):
        for key in args.dataset or dataset_keys():
            rows.extend(run_dataset(key, args.repeats))
    write_json(RESULTS/'hdbscan'/'run_summary.json', rows)
    print(json.dumps(clean_json(dict(stage='complete', datasets=len({x['dataset'] for x in rows}),
                                    finalists=len(rows)))), flush=True)


if __name__ == '__main__':
    main()
