"""Predeclared resolution checkpoints, not claims of optimal K."""
import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from .common import dataset_keys,load_dataset,Evaluator,SEED,RESULTS,finalize

def main():
    for key in dataset_keys():
        ds=load_dataset(key);ev=Evaluator(ds)
        for k in [8,12,16]:
            variant=f'resolution_{k}'
            if (RESULTS/'kmeans'/key/variant/'summary.json').exists():continue
            def fit(x,seed,k=k):return KMeans(n_clusters=k,n_init=20,max_iter=500,tol=1e-4,random_state=seed).fit_predict(x)
            labels=fit(ds.X,SEED)
            finalize(ds,'kmeans',dict(n_clusters=k,n_init=20,max_iter=500,tol=1e-4,random_state=SEED),
                     labels,fit,ev,variant=variant,repeats=30,
                     extras={'selection_rule':'Fixed resolution checkpoint 8/12/16: test stability vs actual profile spread, not force optimal K.'})

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
