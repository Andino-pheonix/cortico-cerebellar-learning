"""Small executable checks on the actual model, using only unittest/NumPy."""
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/distributed'))
from distributed_simulation import DistributedModel
from model_core import rotation,DT,digest
import numpy as np

class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=DistributedModel(11);cls.model.train_cortex()
        cls.base=cls.model.readout.copy()

    def setUp(self):
        self.model.initialize_adaptation(self.base)

    def test_preparation_gate(self):
        _,d=self.model.trial()
        self.assertEqual(len(d['preparation']),10)
        np.testing.assert_array_equal(d['display'][0],np.zeros(2))

    def test_no_future_rotation_before_feedback(self):
        _,a=self.model.trial(theta=0.)
        _,b=self.model.trial(theta=30.)
        np.testing.assert_array_equal(a['hidden'][:12],b['hidden'][:12])
        np.testing.assert_array_equal(a['commands'][:12],b['commands'][:12])

    def test_cortex_is_only_body_output(self):
        _,d=self.model.trial()
        actual=np.vstack([np.zeros(2),np.cumsum(d['commands']*DT,axis=0)])@rotation(30).T
        np.testing.assert_allclose(actual,d['display'],atol=1e-10,rtol=0)

    def test_learning_switches(self):
        self.model.trial(learn_cortex=.125,learn_cb=0.,rate=1.)
        self.assertGreater(np.linalg.norm(self.model.cortical_delta),0)
        np.testing.assert_array_equal(self.model.cerebellar,np.zeros_like(self.model.cerebellar))
        self.model.initialize_adaptation(self.base)
        self.model.trial(learn_cortex=0.,learn_cb=4.26666667,rate=1.)
        self.assertGreater(np.linalg.norm(self.model.cerebellar),0)
        np.testing.assert_array_equal(self.model.cortical_delta,np.zeros_like(self.model.cortical_delta))

    def test_fixed_core_and_teaching_count(self):
        before=digest(np.r_[self.model.recurrent.ravel(),self.model.input.ravel(),self.model.bias.ravel()])
        metric,_=self.model.trial(learn_cortex=.125,learn_cb=4.26666667,rate=1.)
        after=digest(np.r_[self.model.recurrent.ravel(),self.model.input.ravel(),self.model.bias.ravel()])
        self.assertEqual(before,after);self.assertEqual(metric['teaching_segments'],67)
        np.testing.assert_array_equal(self.model.base_readout,self.base)

    def test_probe_freezes_weights_and_removes_only_learned_effect(self):
        self.model.trial(learn_cortex=.125,learn_cb=4.26666667,rate=1.)
        c,b=self.model.cortical_delta.copy(),self.model.cerebellar.copy()
        _,blocked=self.model.trial(connection=False,cortical_memory=False)
        np.testing.assert_array_equal(c,self.model.cortical_delta)
        np.testing.assert_array_equal(b,self.model.cerebellar)
        self.model.initialize_adaptation(self.base);_,base=self.model.trial()
        np.testing.assert_array_equal(blocked['display'],base['display'])

if __name__=='__main__':unittest.main()
