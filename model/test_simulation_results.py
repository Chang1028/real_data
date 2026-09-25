import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/bilinear_mpl')
import matplotlib.pyplot as plt
import numpy as np

from evaluation import predict_bilinear
from simulation_results import (betas_from_blocks, run_simulation_experiment,
                                load_simulation_experiment, plot_saved_betas)


class SavedSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        rng = np.random.default_rng(17)
        x1 = rng.normal(size=(40, 4, 4))
        x1 = (x1 + x1.transpose(0, 2, 1)) / 2
        x2 = rng.uniform(.01, 2, size=(40, 4, 4))
        x2 = (x2 + x2.transpose(0, 2, 1)) / 2
        x2[:, np.arange(4), np.arange(4)] = 3
        np.save(cls.root/'z_fc.npy', x1)
        np.save(cls.root/'z_sc.npy', x2)
        cls.params = dict(DATA_DIR=str(cls.root), P=4, N_SUBJECTS=40, SNR=10,
                          LAMBDA1_GRID=[.01, .02], LAMBDA2_GRID=[.01],
                          N_SPLITS=2, N_JOBS=1, MAX_ITER=1000, NUM_CANDIDATES=1)
        cls.betas = betas_from_blocks(4, [(0, 2, 1.)], [(2, 4, -.7)])
        with contextlib.redirect_stdout(io.StringIO()):
            cls.first = run_simulation_experiment('case', *cls.betas, parameters=cls.params,
                                                 results_dir=cls.root/'results')
            different = betas_from_blocks(4, [(1, 3, 1.2)], [(0, 2, -.5)])
            cls.second = run_simulation_experiment('case', *different, parameters=cls.params,
                                                  results_dir=cls.root/'results')

    @classmethod
    def tearDownClass(cls):
        plt.close('all')
        cls.tmp.cleanup()

    def test_distinct_truths_have_distinct_complete_archives(self):
        a, b = load_simulation_experiment(self.first), load_simulation_experiment(self.second)
        self.assertNotEqual(self.first, self.second)
        self.assertNotEqual(a['metadata']['beta_fingerprint'], b['metadata']['beta_fingerprint'])
        self.assertEqual(a['metadata']['status'], 'complete')
        np.testing.assert_array_equal(a['simulation']['beta1_true'], self.betas[0])
        self.assertNotIn('X1', a['simulation'])
        self.assertTrue((self.first/'source'/'bilinear_lasso.py').exists())
        self.assertEqual(a['metadata']['final_selection'], 'minimum mean CV validation MSE')
        best = a['cv_summary'].sort_values(['Test MSE','lambda1','lambda2']).iloc[0]
        self.assertEqual(a['model']['lambda1'], best['lambda1'])

    def test_final_and_cv_predictions_reconstruct_from_saved_arrays(self):
        s = load_simulation_experiment(self.first, load_connectivity=True)
        model, data = s['model'], s['simulation']
        predicted = predict_bilinear(data['X1']-model['X1_mean'], data['X2']-model['X2_mean'], model['beta1'],model['beta2'])+model['y_mean']
        np.testing.assert_allclose(predicted, model['y_pred'], atol=1e-10)
        for result in s['cv_results']:
            for fold in result['folds']:
                tr,va=fold['train_idx'],fold['val_idx']
                predicted = predict_bilinear(
                    data['X1'][:,:,va]-data['X1'][:,:,tr].mean(2,keepdims=True),
                    data['X2'][:,:,va]-data['X2'][:,:,tr].mean(2,keepdims=True),
                    fold['beta1'],fold['beta2'])
                np.testing.assert_allclose(predicted,fold['y_pred'],atol=1e-10)
                self.assertIn('trajectory',fold['model_history'])
        self.assertIn('stationarity_history',s['history'])

    def test_plotting_notebook_executes_without_fitting(self):
        nb=json.loads((Path(__file__).parent/'plot_saved_simulation.ipynb').read_text())
        ns={}
        with patch('bilinear_lasso.bilinear_lasso.fit',side_effect=AssertionError('Plots must not refit')), contextlib.redirect_stdout(io.StringIO()):
            for i,cell in enumerate(nb['cells']):
                if cell['cell_type']!='code':continue
                exec(compile(''.join(cell['source']),f'plot cell {i}','exec'),ns)
                if i==1: ns['RESULTS_DIR']=self.root/'results'
                plt.close('all')
        self.assertTrue((ns['RUN_DIR']/'figures'/'true_vs_estimated_betas.png').exists())

    def test_final_failure_keeps_cv_results_and_marks_run_failed(self):
        prior=load_simulation_experiment(self.first)
        with patch('simulation_results.run_cv_over_lambdas',return_value=(prior['cv_results'],prior['cv_summary'])), patch('bilinear_lasso.bilinear_lasso.fit',side_effect=RuntimeError('test final failure')), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'partial results'):
                run_simulation_experiment('failed',*self.betas,parameters=self.params,results_dir=self.root/'results',save_connectivity=False)
        failed=next((self.root/'results').glob('failed_*'))
        metadata=json.loads((failed/'metadata.json').read_text())
        self.assertEqual(metadata['status'],'failed')
        self.assertEqual(metadata['stage'],'final_fit')
        self.assertTrue((failed/'cv_results.pkl').exists())
        with self.assertRaisesRegex(ValueError,'not complete'):
            load_simulation_experiment(failed)

    def test_invalid_roi_blocks_are_rejected(self):
        for blocks in ([(0,5,1)],[(0,2,1),(1,3,2)],[(0,2,float('nan'))]):
            with self.assertRaises(ValueError):betas_from_blocks(4,blocks,[])


if __name__=='__main__':
    unittest.main()
