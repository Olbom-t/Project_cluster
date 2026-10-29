"""Regularized Gaussian-mixture pilot on the original, untransformed shares.

BIC is conditional on the regularization and ambient-coordinate model. It is
not evidence of a true number of electoral types. No PCA, log ratios, VIF,
column standardization, or population weights are used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from .common import (SEED, RESULTS, Evaluator, dataset_keys, load_dataset,
                     finalize, write_json, clean_json)


def fit_model(X, k, covariance, seed, reg_covar=1e-6):
    diagnostics = []
    for max_iter in (300, 600):
        model = GaussianMixture(
            n_components=int(k), covariance_type=covariance,
            tol=1e-3, reg_covar=reg_covar, max_iter=max_iter,
            n_init=2, init_params='kmeans', random_state=int(seed),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(X)
        diagnostics.append(dict(max_iter=max_iter, converged=bool(model.converged_),
                                n_iter=int(model.n_iter_),
                                warnings=[str(w.message) for w in caught]))
        if model.converged_:
            return model, diagnostics
    raise RuntimeError(f'GMM did not converge: {diagnostics}')


def process(key, repeats=30):
    ds = load_dataset(key)
    evaluator = Evaluator(ds)
    base = RESULTS/'gmm'/key
    base.mkdir(parents=True, exist_ok=True)
    candidates, cached, errors = [], {}, []
    started = time.time()

    def candidate(k, cov):
        fit_start = time.time()
        try:
            model, diag = fit_model(ds.X, k, cov, SEED+k)
            labels = model.predict(ds.X)
            score = evaluator.score(labels)
            row = dict(dataset=key, covariance_type=cov, n_components=k,
                       reg_covar=1e-6, bic=float(model.bic(ds.X)),
                       aic=float(model.aic(ds.X)), log_likelihood=float(model.score(ds.X)),
                       converged=bool(model.converged_), n_iter=int(model.n_iter_),
                       max_iter_used=diag[-1]['max_iter'],
                       warnings_count=sum(len(x['warnings']) for x in diag),
                       elapsed_seconds=time.time()-fit_start, **score)
            candidates.append(row)
            cached[(cov,k)] = (model,labels,diag)
        except Exception as e:
            errors.append(dict(dataset=key, covariance_type=cov,n_components=k,error=repr(e)))
        pd.DataFrame(candidates).to_csv(base/'candidates.csv',index=False)
        write_json(base/'fit_errors.json',errors)

    for cov in ('full','diag'):
        for k in range(2,16):
            candidate(k,cov)
        family = [r for r in candidates if r['covariance_type']==cov]
        if not family:
            continue
        if min(family,key=lambda r:r['bic'])['n_components']==15:
            for k in range(16,21):
                candidate(k,cov)

    selected = []
    for cov in ('full','diag'):
        family = [r for r in candidates if r['covariance_type']==cov]
        if not family:
            continue
        best = min(family,key=lambda r:r['bic'])
        k = int(best['n_components'])
        model,labels,diag = cached[(cov,k)]
        resampling_diagnostics = []

        def fit_predict(X, seed, selected_k=k, selected_cov=cov):
            fitted, diagnostic = fit_model(X,selected_k,selected_cov,seed)
            resampling_diagnostics.append(dict(seed=seed,attempts=diagnostic))
            return fitted.predict(X)

        sensitivity = {}
        try:
            sensitive, sensitive_diag = fit_model(ds.X,k,cov,SEED+k,reg_covar=1e-5)
            sensitivity = dict(reg_covar=1e-5,
                               ari_against_primary=float(adjusted_rand_score(labels,sensitive.predict(ds.X))),
                               metrics=evaluator.score(sensitive.predict(ds.X)),
                               bic=float(sensitive.bic(ds.X)),aic=float(sensitive.aic(ds.X)),
                               diagnostics=sensitive_diag)
        except Exception as e:
            sensitivity = dict(reg_covar=1e-5,error=repr(e))

        variant = 'bic_'+cov
        extras = dict(selection='Lowest BIC within this covariance family',
                      selection_row=best,grid_components=[r['n_components'] for r in family],
                      selection_at_upper_boundary=k==max(r['n_components'] for r in family),
                      fit_diagnostics=diag,resampling_fit_diagnostics=resampling_diagnostics,
                      regularization_sensitivity=sensitivity,
                      interpretation='Gaussian component probabilities are conditional model probabilities, not validated probabilities of true electoral types.',
                      limitation='The original shares sum to one. reg_covar regularizes singular covariance matrices without changing the input shares. Ambient-coordinate BIC depends on this model specification; full/diag BIC values are not independent evidence of true cluster count.')
        params=dict(n_components=k,covariance_type=cov,reg_covar=1e-6,
                    n_init=2,max_iter=300,retry_max_iter=600,tol=1e-3,
                    init_params='kmeans',random_state=SEED+k)
        info=finalize(ds,'gmm',params,labels,fit_predict,evaluator,
                      variant=variant,extras=extras,repeats=repeats)

        probabilities=model.predict_proba(ds.X)
        order=np.argsort(probabilities,axis=1)
        id_map={label:f'C{i+1:02d}' for i,label in enumerate(sorted(set(labels)))}
        assignments=pd.read_csv(base/variant/'assignments.csv',dtype={'uik':str})
        assignments['gmm_max_posterior']=probabilities[np.arange(len(ds.X)),order[:,-1]]
        assignments['gmm_second_posterior']=probabilities[np.arange(len(ds.X)),order[:,-2]]
        assignments['gmm_alternative_cluster']=[id_map.get(int(i),'unused_component') for i in order[:,-2]]
        assignments.to_csv(base/variant/'assignments.csv',index=False,encoding='utf-8-sig')
        selected.append(info)

    write_json(base/'run_status.json',dict(dataset=key,completed=True,
               candidate_count=len(candidates),fit_errors=errors,
               elapsed_seconds=time.time()-started,selected_variants=[x['variant'] for x in selected]))
    return selected


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='*')
    parser.add_argument('--repeats',type=int,default=30)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    keys=args.datasets or dataset_keys()
    with threadpool_limits(limits=1):
        for key in keys:
            done=RESULTS/'gmm'/key/'run_status.json'
            if args.resume and done.exists() and json.loads(done.read_text()).get('completed'):
                print(f'Reusing completed GMM run: {key}',flush=True)
                continue
            print(f'Starting GMM: {key}',flush=True)
            process(key,args.repeats)


if __name__=='__main__':
    main()
