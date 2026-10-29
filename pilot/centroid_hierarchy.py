"""K-means and Ward pilot; fixed features, equal precinct weights."""
import argparse
import time
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage,cut_tree
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from .common import dataset_keys,load_dataset,Evaluator,RESULTS,SEED,write_json,finalize

def choose_candidates(rows, base_sse):
    df=pd.DataFrame(rows).sort_values('k')
    good=df[df.silhouette.notna()]
    if good.empty:raise ValueError('No valid multi-cluster candidate')
    silhouette_k=int(good.sort_values(['silhouette','k'],ascending=[False,True]).iloc[0].k)
    # Maximum departure below straight line between K=1 and last evaluated K.
    # This is a transparent descriptive elbow heuristic, not a significance test.
    xs=(df.k.to_numpy()-1)/(df.k.max()-1)
    ys=(base_sse-df.sse.to_numpy())/(base_sse-df.sse.iloc[-1])
    df['elbow_gap']=ys-xs
    knee_k=int(df.iloc[int(np.argmax(df.elbow_gap))].k)
    return df,dict(silhouette= silhouette_k,elbow=knee_k)

def run(method, keys=None, repeats=30):
    for key in keys or dataset_keys():
        ds=load_dataset(key);ev=Evaluator(ds);start=time.time();out=RESULTS/method/key;out.mkdir(parents=True,exist_ok=True)
        labels_by_k={};rows=[];base_sse=float(((ds.X-ds.X.mean(axis=0))**2).sum())
        Z=linkage(ds.X,method='ward') if method=='ward' else None
        def fit(x,k,seed):
            if method=='kmeans':
                return KMeans(n_clusters=k,n_init=20,max_iter=500,tol=1e-4,random_state=seed,algorithm='lloyd').fit_predict(x)
            return cut_tree(linkage(x,method='ward'),n_clusters=[k]).ravel()
        for k in range(2,21):
            labels=cut_tree(Z,n_clusters=[k]).ravel() if Z is not None else fit(ds.X,k,SEED)
            labels_by_k[k]=labels;row=ev.score(labels);row['requested_k']=k;row['sse_fraction_of_k1']=row['sse']/base_sse;rows.append(row)
        if max(rows,key=lambda r:r['silhouette'])['k']==20:
            for k in range(21,31):
                labels=cut_tree(Z,n_clusters=[k]).ravel() if Z is not None else fit(ds.X,k,SEED)
                labels_by_k[k]=labels;row=ev.score(labels);row['requested_k']=k;row['sse_fraction_of_k1']=row['sse']/base_sse;rows.append(row)
        frame,candidates=choose_candidates(rows,base_sse);frame.to_csv(out/'candidates.csv',index=False)
        write_json(out/'selection.json',dict(rules={'silhouette':'Maximum silhouette on fixed common sample',
                                                  'elbow':'Maximum normalized SSE departure from K=1 to last searched K; not a formal optimum'},
                                           candidates=candidates,base_sse=base_sse,grid_seconds=time.time()-start))
        done={}
        for variant,k in candidates.items():
            if k in done:
                write_json(out/f'{variant}_alias.json',{'same_as':done[k],'k':k});continue
            params=dict(n_clusters=k,metric='euclidean',features='all named raw vote shares',equal_precinct_weights=True)
            if method=='kmeans':params.update(n_init=20,max_iter=500,tol=1e-4,random_state=SEED)
            else:params.update(linkage='ward')
            finalize(ds,method,params,labels_by_k[k],lambda x,seed,k=k:fit(x,k,seed),ev,variant=variant,repeats=repeats,
                     extras=dict(selection_rule=variant,base_sse=base_sse,explained_sse_fraction=1-ev.score(labels_by_k[k])['sse']/base_sse))
            done[k]=variant

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['kmeans','ward'],required=True)
    parser.add_argument('--keys',nargs='*');parser.add_argument('--repeats',type=int,default=30);args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.method,args.keys,args.repeats)
