"""有限ループの再開・失敗判定。実学習を起動しない。"""
import unittest
from run_relative_sequence_cycle import completed_stage, compare_contracts


class SequenceCycleTest(unittest.TestCase):
    def test_completed_receipt_and_output_required(self):
        good = {'returncode': 0, 'live_descendants': []}
        self.assertTrue(completed_stage(good, {'completed': True}, training=False))
        for receipt in ({'returncode': 1}, {'returncode': 0, 'live_descendants': [1]}):
            with self.assertRaises(ValueError): completed_stage(receipt, {'completed': True}, training=False)
        with self.assertRaises(ValueError): completed_stage(good, None, training=False)

    def test_partial_training_does_not_advance_to_scale_comparison(self):
        good = {'returncode': 0, 'live_descendants': []}
        with self.assertRaises(ValueError):
            completed_stage(good, {'completed': True, 'fixed_updates_completed': False}, training=True)
        self.assertTrue(completed_stage(good, {'completed': True, 'fixed_updates_completed': True}, training=True))

    def test_same_scales_or_changed_fixed_factors_rejected(self):
        keys = ('initial_model_sha256', 'initial_rng_sha256', 'samples_sha256', 'labels_sha256',
                'batches_sha256', 'source_sha256', 'helper_sha256', 'official_update_source_sha256',
                'updates', 'batch_size', 'learning_rate', 'warmup', 'grad_clip', 'padding_loss', 'time_sampling')
        a = {key: 'same' for key in keys};a['scale']='old'
        b = dict(a,scale='relative')
        self.assertTrue(compare_contracts(a,b))
        for key in keys:
            with self.assertRaises(ValueError): compare_contracts(a,dict(b,**{key:'different'}))
        with self.assertRaises(ValueError): compare_contracts(a,a)


if __name__=='__main__':unittest.main()
