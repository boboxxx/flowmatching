import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('ack_risk', Path(__file__).resolve().parents[1]/'scripts/audit_ack_risk.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def case(psnr=23, prediction=24):
    base = dict(psnr=str(psnr), lpips='.2', ssim='.8',
                predicted_direct_psnr_raw=str(prediction), predicted_fm_psnr_raw=str(prediction))
    return {'direct':base, 'fm_only':base, 'full_harq':dict(base, psnr='26')}


class AckRiskTest(unittest.TestCase):
    def test_zero_coverage_is_not_zero_conditional_risk(self):
        result = audit.evaluate({0:case()}, 'flowharq', None, 0)
        self.assertIsNone(result['conditional_false_ack'])
        self.assertEqual(result['ack_coverage'], 0)
        self.assertEqual(result['charged_uses'], 16388)

    def test_false_ack_at_threshold(self):
        result = audit.evaluate({0:case()}, 'adaptive_harq', 0, 0)
        self.assertEqual(result['conditional_false_ack'], 1)
        self.assertEqual(result['joint_false_ack'], 1)
        self.assertEqual(result['charged_uses'], 12290)

    def test_empirical_selection_and_fallback(self):
        curve = [audit.evaluate({0:case(25,25)}, 'flowharq', m, 0) for m in audit.MARGINS]
        self.assertEqual(audit.select(curve)['margin_db'], 0)
        failed = [audit.evaluate({0:case()}, 'flowharq', m, 0) for m in audit.MARGINS]
        self.assertEqual(audit.select(failed)['status'], 'infeasible_nontrivial_policy')

    def test_mse_is_mean_of_per_image_values(self):
        result = audit.evaluate({0:case(20,30), 1:case(30,30)}, 'flowharq', 0, 0)
        self.assertAlmostEqual(result['mse'], .0055)
        self.assertNotAlmostEqual(result['mse'], 10**(-result['psnr']/10))

    def test_margin_monotonicity(self):
        cases = {0:case(23,24.1), 1:case(25,26)}
        values = [audit.evaluate(cases, 'flowharq', m, 0)['nack'] for m in audit.MARGINS]
        self.assertEqual(values, sorted(values))


if __name__ == '__main__':
    unittest.main()
