"""同じ画像の乱数差と平均誤差を、別の指標として集計する。"""
import unittest
import numpy as np
from analyze_relative_sequence import error_components, matching_rows


class SequenceResultAnalysisTest(unittest.TestCase):
    def test_mse_decomposition_keeps_bias_and_variation_separate(self):
        error=np.full((5,2,8),.002)
        error[:,:,0]+=np.array([-2,-1,0,1,2])[:,None]*.001
        result=error_components(error)
        self.assertAlmostEqual(result['joint_rmse_rad']**2,
            result['mean_prediction_joint_rmse_rad']**2+result['seed_residual_joint_rms_rad']**2)
        self.assertAlmostEqual(result['mean_prediction_joint_rmse_rad'],.002)
        self.assertAlmostEqual(result['mean_prediction_gap_max_mm'],2.)

    def test_identical_outputs_have_no_seed_variation(self):
        result=error_components(np.zeros((5,3,8)))
        self.assertEqual(result['seed_residual_joint_rms_rad'],0.)
        self.assertEqual(result['mean_error_fraction'],0.)

    def test_shapes_and_nonfinite_rejected(self):
        for error in (np.zeros((4,2,8)),np.zeros((5,0,8)),np.zeros((5,2,7)),np.full((5,2,8),np.nan)):
            with self.assertRaises(ValueError):error_components(error)

    def test_only_prediction_can_differ(self):
        a=[dict(index=0,actual_open=1.,reference=[0.]*8,prediction=[0.]*8)]
        b=[dict(a[0],prediction=[1.]*8)]
        self.assertTrue(matching_rows(a,b))
        for changed in (dict(b[0],index=1),dict(b[0],actual_open=0.),dict(b[0],reference=[1.]*8)):
            with self.assertRaises(ValueError):matching_rows(a,[changed])
        with self.assertRaises(ValueError):matching_rows(a,[])


if __name__=='__main__':unittest.main()
