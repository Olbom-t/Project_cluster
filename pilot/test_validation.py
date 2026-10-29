import unittest
import numpy as np
import pandas as pd
from .common import Dataset,Evaluator,stability,load_dataset,dataset_keys

class PilotValidation(unittest.TestCase):
    def test_permuted_labels_and_noise_do_not_break_stability(self):
        X=np.vstack([np.tile([.9,.1],(20,1)),np.tile([.1,.9],(20,1)),np.tile([.5,.5],(10,1))])
        ds=Dataset('test','test',0,X,['a','b'],pd.DataFrame({'uik':np.arange(50)}))
        ref=np.array([0]*20+[1]*20+[-1]*10)
        def fit(x,seed):return np.where(x[:,0]>.7,42,np.where(x[:,0]<.3,7,-1))
        summary,points,_=stability(ds,ref,fit,repeats=12)
        self.assertEqual(summary['ari_all_median'],1.)
        self.assertEqual(summary['ari_common_median'],1.)
        self.assertEqual(summary['min_cluster_jaccard_median'],1.)
        np.testing.assert_array_equal(points.assignment_stability.to_numpy(),np.ones(len(X)))
        self.assertTrue((points.noise_frequency.iloc[40:]==1).all())
        score=Evaluator(ds).score(ref)
        self.assertEqual(score['coverage'],.8)
        self.assertAlmostEqual(score['silhouette'],1.)
    def test_all_prepared_data_have_valid_equal_weight_features(self):
        keys=dataset_keys();self.assertEqual(len(keys),12);n=0
        for key in keys:
            ds=load_dataset(key);n+=len(ds.X)
            self.assertTrue(np.isfinite(ds.X).all())
            self.assertTrue((ds.X>=0).all());self.assertTrue((ds.meta.valid>0).all())
            np.testing.assert_allclose(ds.X.sum(axis=1),1.,atol=1e-12)
            self.assertEqual(ds.X.shape[1],14 if ds.year==2021 else 4)
            self.assertFalse(ds.meta.uik.isna().any())
        self.assertEqual(n,29743)

if __name__=='__main__':unittest.main()
