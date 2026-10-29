"""Small exploratory repetition of K selection, distinct from fixed-K stability."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances,silhouette_score
from threadpoolctl import threadpool_limits
from .common import dataset_keys,load_dataset,RESULTS,SEED,write_json

def main(repeats=10):
    for key in dataset_keys():
        ds=load_dataset(key);out=RESULTS/'k_selection'/key;out.mkdir(parents=True,exist_ok=True)
        if (out/'summary.json').exists():continue
        rng=np.random.default_rng(SEED+100000);records=[]
        for b in range(repeats):
            idx=np.sort(rng.choice(len(ds.X),int(.8*len(ds.X)),replace=False));X=ds.X[idx]
            sample=np.sort(rng.choice(len(X),min(1000,len(X)),replace=False))
            D=pairwise_distances(X[sample]);np.fill_diagonal(D,0.)
            base=float(((X-X.mean(axis=0))**2).sum());rows=[]
            fit_seed=int(rng.integers(2**31-1))
            for k in range(2,21):
                model=KMeans(n_clusters=k,n_init=20,max_iter=500,tol=1e-4,random_state=fit_seed).fit(X)
                labels=model.labels_[sample]
                sil=float(silhouette_score(D,labels,metric='precomputed')) if len(np.unique(labels))>1 else np.nan
                rows.append((k,model.inertia_,sil))
            arr=np.asarray(rows);gap=(base-arr[:,1])/(base-arr[-1,1])-(arr[:,0]-1)/19
            knee=int(arr[np.argmax(gap),0]);best=int(arr[np.nanargmax(arr[:,2]),0])
            records.append(dict(repeat=b,elbow_k=knee,silhouette_k=best))
        frame=pd.DataFrame(records);frame.to_csv(out/'repetitions.csv',index=False)
        summary=dict(dataset=key,repeats=repeats,fraction=.8,grid=[2,20],n_init=20,silhouette_sample=1000,
                     elbow_counts=frame.elbow_k.value_counts().to_dict(),silhouette_counts=frame.silhouette_k.value_counts().to_dict(),
                     note='Exploratory K-selection variability; 10 repetitions, not external validation; silhouette sample differs from main 1500-point evaluation.')
        write_json(out/'summary.json',summary);print(key,summary['elbow_counts'],flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
