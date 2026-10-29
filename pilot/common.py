from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_score, pairwise_distances

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
SEED = 20261028

def clean_json(x):
    if isinstance(x, dict): return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray): return clean_json(x.tolist())
    if isinstance(x, np.generic): return clean_json(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x

def write_json(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean_json(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

@dataclass
class Dataset:
    key: str
    region: str
    year: int
    X: np.ndarray
    features: list[str]
    meta: pd.DataFrame

def dataset_keys(): return sorted(p.stem for p in DATA.glob('*.npz'))

def load_dataset(key):
    with np.load(DATA/f'{key}.npz',allow_pickle=False) as z:
        return Dataset(key,str(z['region']),int(z['year']),z['X'],z['features'].tolist(),
                       pd.read_csv(DATA/f'{key}.csv',dtype={'uik':str}))

def normalize_labels(labels):
    labels=np.asarray(labels,dtype=int)
    values=sorted(set(labels)-{-1})
    return np.array([values.index(int(v)) if v>=0 else -1 for v in labels])

class Evaluator:
    """Same fixed uniform sample for all candidates of a dataset; noise excluded only from silhouette."""
    def __init__(self, ds, sample_size=1500):
        self.ds=ds
        seed=SEED+int(hashlib.sha256(ds.key.encode()).hexdigest()[:6],16)
        self.sample=np.sort(np.random.default_rng(seed).choice(len(ds.X),min(sample_size,len(ds.X)),replace=False))
        self.dist=pairwise_distances(ds.X[self.sample],metric='euclidean')
        np.fill_diagonal(self.dist,0)

    def score(self, labels):
        labels=normalize_labels(labels); groups=sorted(set(labels)-{-1})
        mask=labels>=0; sample_labels=labels[self.sample];smask=sample_labels>=0
        unique=np.unique(sample_labels[smask]);sil=np.nan
        if 1<len(unique)<int(smask.sum()):
            sil=float(silhouette_score(self.dist[np.ix_(smask,smask)],sample_labels[smask],metric='precomputed'))
        sizes=[];sse=0.; ranges=[];q90ranges=[]
        for k in groups:
            x=self.ds.X[labels==k];sizes.append(len(x));sse+=float(((x-x.mean(axis=0))**2).sum())
            ranges.append(float(np.ptp(x,axis=0).max()*100))
            q90ranges.append(float((np.quantile(x,.95,axis=0)-np.quantile(x,.05,axis=0)).max()*100))
        return dict(n=len(labels),k=len(groups),coverage=float(mask.mean()),noise_count=int((~mask).sum()),
                    silhouette=sil,silhouette_sample_assigned=int(smask.sum()),silhouette_sample_size=len(self.sample),
                    min_cluster_size=min(sizes) if sizes else 0,max_cluster_size=max(sizes) if sizes else 0,
                    clusters_lt_10=sum(s<10 for s in sizes),sse=sse,
                    rmse_profile_pp=100*np.sqrt(sse/mask.sum()) if mask.any() else np.nan,
                    max_party_range_pp=max(ranges) if ranges else np.nan,
                    max_party_central90_range_pp=max(q90ranges) if q90ranges else np.nan)

def stability(ds, reference_labels, fit_predict, repeats=30, fraction=.8, seed=SEED):
    """Descriptive internal resampling stability, evaluated only on shared sampled observations.

    No held-out predictive claim. Each repetition refits on a uniform 80% subsample.
    Noise is excluded from cluster matching; noise transitions remain in per-object denominators.
    """
    ref=normalize_labels(reference_labels);n=len(ref);K=len(set(ref)-{-1})
    counts=np.zeros((n,K+2),dtype=int);seen=np.zeros(n,dtype=int)
    records=[];recoveries={k:[] for k in range(K)};failures=[]
    rng=np.random.default_rng(seed)
    for b in range(repeats):
        idx=np.sort(rng.choice(n,max(2,int(np.floor(fraction*n))),replace=False))
        try:
            lab=normalize_labels(fit_predict(ds.X[idx],int(rng.integers(0,2**31-1))))
            if len(lab)!=len(idx):raise ValueError('Wrong label length')
            ri=ref[idx];groups=sorted(set(lab)-{-1});jac=np.zeros((K,len(groups)))
            for k in range(K):
                for t,g in enumerate(groups):
                    a=ri==k;c=lab==g;union=np.sum(a|c)
                    jac[k,t]=np.sum(a&c)/union if union else 0.
            mapping={}
            if jac.size:
                rows,cols=linear_sum_assignment(-jac)
                mapping={groups[c]:int(r) for r,c in zip(rows,cols) if jac[r,c]>0}
            aligned=np.array([mapping.get(int(v),K+1) if v>=0 else K for v in lab])
            counts[idx,aligned]+=1;seen[idx]+=1
            common=(ri>=0)&(lab>=0)
            ari_common=float(adjusted_rand_score(ri[common],lab[common])) if len(np.unique(ri[common]))>1 and len(np.unique(lab[common]))>1 else np.nan
            for k in range(K):
                if np.any(ri==k):
                    g=next((g for g,r in mapping.items() if r==k),None)
                    recoveries[k].append(float(jac[k,groups.index(g)]) if g is not None else 0.)
            records.append(dict(repeat=b,k=len(groups),coverage=float((lab>=0).mean()),
                                ari_all=float(adjusted_rand_score(ri,lab)),ari_common_assigned=ari_common,
                                common_assigned_fraction=float(common.mean())))
        except Exception as e:
            failures.append(dict(repeat=b,error=repr(e)))
    point=pd.DataFrame({'resample_appearances':seen})
    own=np.full(n,np.nan)
    for i in range(n):
        if seen[i]>0:own[i]=counts[i,ref[i] if ref[i]>=0 else K]/seen[i]
    point['assignment_stability']=own
    point['noise_frequency']=np.divide(counts[:,K],seen,out=np.full(n,np.nan),where=seen>0)
    point['unmatched_cluster_frequency']=np.divide(counts[:,K+1],seen,out=np.full(n,np.nan),where=seen>0)
    alt=[];freq=[]
    for i in range(n):
        row=counts[i,:K].copy()
        if ref[i]>=0:row[ref[i]]=0
        winner=int(np.argmax(row)) if K else -1
        alt.append(f'C{winner+1:02d}' if winner>=0 and row[winner]>0 else '')
        freq.append(float(row[winner]/seen[i]) if winner>=0 and seen[i] else 0.)
    point['resample_alternative_cluster']=alt;point['resample_alternative_frequency']=freq
    rec=pd.DataFrame(records)
    def median(col):return float(rec[col].median()) if len(rec) else np.nan
    cluster_j={str(k):float(np.median(v)) if v else np.nan for k,v in recoveries.items()}
    summary=dict(repeats_requested=repeats,repeats_successful=len(records),failures=failures,fraction=fraction,
                 ari_all_median=median('ari_all'),ari_common_median=median('ari_common_assigned'),
                 median_resampled_coverage=median('coverage'),median_resampled_k=median('k'),
                 resampled_k_counts=rec.k.value_counts().to_dict() if len(rec) else {},
                 cluster_jaccard_medians=cluster_j,min_cluster_jaccard_median=min(cluster_j.values()) if cluster_j else np.nan,
                 unstable_assigned_fraction=float(np.mean(own[ref>=0]<.8)) if K else np.nan,
                 point_stability_median=float(np.nanmedian(own[ref>=0])) if K else np.nan,
                 note='Internal stability on shared in-sample observations; no independent validation or true-class probability.')
    return summary,point,rec

def finalize(ds, method, params, labels, fit_predict, evaluator, variant='selected', extras=None, repeats=30):
    labels=normalize_labels(labels);base=RESULTS/method/ds.key/variant;base.mkdir(parents=True,exist_ok=True)
    start=time.time();summary,points,runs=stability(ds,labels,fit_predict,repeats=repeats)
    score=evaluator.score(labels)
    info=dict(dataset=ds.key,region=ds.region,year=ds.year,method=method,variant=variant,
              parameters=params,metrics=score,stability=summary,extras=extras or {},elapsed_stability_seconds=time.time()-start)
    write_json(base/'summary.json',info);runs.to_csv(base/'resampling.csv',index=False)
    result=ds.meta.copy();result['cluster']=[f'C{k+1:02d}' if k>=0 else 'unassigned' for k in labels]
    result=pd.concat([result.reset_index(drop=True),points],axis=1)
    profiles=[];groups=sorted(set(labels)-{-1})
    for k in groups:
        mask=labels==k;x=ds.X[mask];center=x.mean(axis=0);dist=np.linalg.norm(x-center,axis=1)
        typical_idx=np.flatnonzero(mask)[int(np.argmin(dist))]
        for j,name in enumerate(ds.features):
            profiles.append(dict(cluster=f'C{k+1:02d}',n=int(mask.sum()),feature=name,mean_pct=float(100*center[j]),
                                 q05_pct=float(100*np.quantile(x[:,j],.05)),q95_pct=float(100*np.quantile(x[:,j],.95)),
                                 min_pct=float(100*x[:,j].min()),max_pct=float(100*x[:,j].max()),
                                 typical_uik=str(ds.meta.iloc[typical_idx].uik)))
    result.to_csv(base/'assignments.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(profiles).to_csv(base/'profiles.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(clean_json(dict(dataset=ds.key,method=method,variant=variant,k=score['k'],coverage=score['coverage'],
                                    silhouette=score['silhouette'],ari=summary['ari_common_median'],min_j=summary['min_cluster_jaccard_median'])),ensure_ascii=False),flush=True)
    return info
