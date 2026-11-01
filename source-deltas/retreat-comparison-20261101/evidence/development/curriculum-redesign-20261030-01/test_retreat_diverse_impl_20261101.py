import hashlib
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
BASELINE_SHA = 'c71001eb26bdf76ed672c39b1ee7d656becdaabc832edc080903f66479d49ead'


class DiagnosticImplementationTests(unittest.TestCase):
    def test_only_registered_changes_to_prior_evaluator(self):
        raw = (HERE / 'evaluate_retreat_ed_impl_20261101.py').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), BASELINE_SHA)
        source = raw.decode('utf8').replace('\r\n', '\n')
        replacements = [
            ('"""開発済みの追加人物で位置回帰２候補の動作を診断する。"""', '"""未採用の退避候補を開発条件で診断する。"""'),
            ('from retreat_coarse_context_20261101 import coarse_context', 'from retreat_diverse_runtime_20261101 import encode_runtime_state'),
            ("parser.add_argument('--context-mode',choices=('control','candidate'),required=True)",
             "parser.add_argument('--context-mode',choices=('coarse',),required=True)"),
            ("retreat_train = BASE / 'dual-host-retreat-ed-candidate-20261101-01'",
             "retreat_train = BASE / 'dual-host-retreat-diverse-coarse-20261101-01'"),
            ("                    context = coarse_context(state,anchors)[16:30]\n"
             "                    retreat_input_state = model_state(state,context,retreat_means,use_context=ARGS.context_mode=='candidate')",
             "                    retreat_input_state = encode_runtime_state(state,anchors,retreat_norm['retreat_context_contract'])"),
        ]
        for old, new in replacements:
            self.assertEqual(source.count(old), 1)
            source = source.replace(old, new, 1)
        actual = (HERE / 'evaluate_retreat_diverse_impl_20261101.py').read_text(encoding='utf8')
        self.assertEqual(actual, source)


if __name__ == '__main__':
    unittest.main()
