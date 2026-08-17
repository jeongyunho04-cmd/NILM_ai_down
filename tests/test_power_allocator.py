"""
Unit Tests for PowerConservationAllocator (V3)
"""
import unittest
import numpy as np
from src.power_allocator import PowerConservationAllocator


class TestPowerConservationAllocatorV3(unittest.TestCase):
    def setUp(self):
        self.allocator = PowerConservationAllocator(
            unknown_appliance_threshold_w=20.0,
            anomaly_duration_sec=0.1, # 테스트용 짧은 지속시간
            hard_gating_threshold=0.25,
            enable_residual_compensation=True
        )

    def test_hard_gating(self):
        """Ghost Leakage 차단: prob < 0.25일 때 0W 클램핑 테스트"""
        p_total = 100.0
        p_preds = {"kettle": 10.0, "fan": 30.0, "beam_projector": 40.0, "laptop_charger": 0.0, "minipc": 0.0}
        state_probs = {"kettle": 0.05, "fan": 0.95, "beam_projector": 0.90, "laptop_charger": 0.0, "minipc": 0.0}
        
        p_final, p_other, _ = self.allocator.allocate_single(p_total, p_preds, state_probs=state_probs)
        
        # kettle은 prob 0.05이므로 0W로 강제 클램핑되어야 함
        self.assertEqual(p_final["kettle"], 0.0)
        self.assertAlmostEqual(sum(p_final.values()) + p_other, p_total, places=4)

    def test_residual_compensation(self):
        """Intelligent Residual Compensation: 빔프로젝터 과소추정 시 Other Power에서 보정되는지 테스트"""
        p_total = 50.0
        p_preds = {"kettle": 0.0, "fan": 0.0, "beam_projector": 35.0, "laptop_charger": 0.0, "minipc": 0.0}
        state_probs = {"kettle": 0.0, "fan": 0.0, "beam_projector": 0.99, "laptop_charger": 0.0, "minipc": 0.0}
        
        p_final, p_other, _ = self.allocator.allocate_single(p_total, p_preds, state_probs=state_probs)
        
        # 빔프로젝터(정격 48W)가 35W에서 48W 부근으로 보정되어야 함
        self.assertGreater(p_final["beam_projector"], 45.0)
        self.assertAlmostEqual(sum(p_final.values()) + p_other, p_total, places=4)

    def test_batch_smoothing_and_conservation(self):
        """시계열 배치 적응형 평활 및 100% 항등식 보존 테스트"""
        N = 100
        p_total_arr = np.full(N, 100.0)
        p_pred_mat = np.zeros((N, 5))
        p_pred_mat[:, 1] = 30.0 + np.random.normal(0, 1.0, N) # Fan + Jitter
        p_pred_mat[:, 2] = 40.0 # Beam
        state_probs = np.zeros((N, 5))
        state_probs[:, 1] = 0.99
        state_probs[:, 2] = 0.99
        
        p_final_mat, p_other_arr, is_unknown = self.allocator.allocate_batch(
            p_total_arr, p_pred_mat, state_probs_matrix=state_probs, sampling_hz=60.0
        )
        
        # 100% 항등식 검증
        total_reconstructed = np.sum(p_final_mat, axis=-1) + p_other_arr
        np.testing.assert_allclose(total_reconstructed, p_total_arr, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
