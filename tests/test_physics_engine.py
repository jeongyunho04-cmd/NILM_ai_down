"""
Unit Tests for ACPhysicsEngine (Parallel Superposition, Harmonic Vectors & Voltage Sag)
"""
import unittest
import numpy as np
from src.ac_physics_engine import ACPhysicsEngine


class TestACPhysicsEngine(unittest.TestCase):

    def setUp(self):
        self.engine = ACPhysicsEngine(
            r_line=0.25,
            l_line=0.15e-3,
            grid_v_nominal=220.0,
            grid_f_nominal=60.0,
            num_harmonics=15
        )

    def test_resistive_loads_superposition(self):
        """두 개의 순수 저항 부하 병렬 중첩 테스트"""
        T = 10
        # Load 1: 1000W, I1 = 4.545A, Phase = 0 deg
        load1 = {
            'p_w': np.full(T, 1000.0),
            'ih_rms': np.zeros((T, 15)),
            'ih_deg': np.zeros((T, 15)),
        }
        load1['ih_rms'][:, 0] = 4.545
        
        # Load 2: 500W, I1 = 2.273A, Phase = 0 deg
        load2 = {
            'p_w': np.full(T, 500.0),
            'ih_rms': np.zeros((T, 15)),
            'ih_deg': np.zeros((T, 15)),
        }
        load2['ih_rms'][:, 0] = 2.273
        
        result = self.engine.superimpose_parallel_loads(
            [load1, load2], apply_voltage_sag=False
        )
        
        # Total P = 1500W
        np.testing.assert_allclose(result['p_w'], 1500.0, rtol=1e-3)
        # Total I1 = 4.545 + 2.273 = 6.818A
        np.testing.assert_allclose(result['ih_rms'][:, 0], 6.818, rtol=1e-3)
        # Total Phase = 0 deg
        np.testing.assert_allclose(result['ih_deg'][:, 0], 0.0, atol=1e-3)
        # Power factor = 1.0 (pure resistive)
        np.testing.assert_allclose(result['power_factor'], 1.0, atol=1e-3)

    def test_orthogonal_phasor_vector_sum(self):
        """저항 부하 (0도) + 순수 유도 부하 (-90도) 직교 벡터합 테스트 (3-4-5 삼각형)"""
        T = 5
        # Load R: I1 = 3.0A, Phase = 0 deg, P = 660W (at 220V)
        load_r = {
            'p_w': np.full(T, 660.0),
            'ih_rms': np.zeros((T, 15)),
            'ih_deg': np.zeros((T, 15)),
        }
        load_r['ih_rms'][:, 0] = 3.0
        load_r['ih_deg'][:, 0] = 0.0
        
        # Load L: I1 = 4.0A, Phase = -90 deg, P = 0W (pure inductive)
        load_l = {
            'p_w': np.full(T, 0.0),
            'ih_rms': np.zeros((T, 15)),
            'ih_deg': np.zeros((T, 15)),
        }
        load_l['ih_rms'][:, 0] = 4.0
        load_l['ih_deg'][:, 0] = -90.0 # Lagging 90 deg
        
        result = self.engine.superimpose_parallel_loads(
            [load_r, load_l], apply_voltage_sag=False
        )
        
        # Total Irms = sqrt(3^2 + 4^2) = 5.0A
        np.testing.assert_allclose(result['irms'], 5.0, rtol=1e-3)
        # Total Phase = atan2(-4, 3) = -53.13 deg
        expected_deg = np.degrees(np.arctan2(-4.0, 3.0)) # ~ -53.1301 deg
        np.testing.assert_allclose(result['ih_deg'][:, 0], expected_deg, atol=1e-2)
        # Total S = 220 * 5 = 1100 VA
        np.testing.assert_allclose(result['s_va'], 1100.0, rtol=1e-3)
        # Power Factor = 660 / 1100 = 0.60
        np.testing.assert_allclose(result['power_factor'], 0.60, atol=1e-3)

    def test_harmonic_cancellation_and_addition(self):
        """고조파 위상 간섭 (상쇄 및 보강) 검증"""
        T = 5
        # Load A: 3차 고조파 2.0A, 위상 0도
        load_a = {
            'p_w': np.full(T, 100.0),
            'ih_rms': np.zeros((T, 15)),
            'ih_deg': np.zeros((T, 15)),
        }
        load_a['ih_rms'][:, 0] = 1.0 # 1차
        load_a['ih_rms'][:, 2] = 2.0 # 3차 (index 2)
        load_a['ih_deg'][:, 2] = 0.0
        
        # Load B: 3차 고조파 2.0A, 위상 180도 (완전 역위상 -> 상쇄)
        load_b = {
            'p_w': np.full(T, 100.0),
            'ih_rms': np.zeros((T, 15)),
            'ih_deg': np.zeros((T, 15)),
        }
        load_b['ih_rms'][:, 0] = 1.0 # 1차
        load_b['ih_rms'][:, 2] = 2.0 # 3차 (index 2)
        load_b['ih_deg'][:, 2] = 180.0
        
        result_cancel = self.engine.superimpose_parallel_loads(
            [load_a, load_b], apply_voltage_sag=False
        )
        
        # 3차 고조파가 상쇄되어 0이어야 함
        np.testing.assert_allclose(result_cancel['ih_rms'][:, 2], 0.0, atol=1e-5)

    def test_line_impedance_voltage_sag(self):
        """선로 임피던스에 의한 대부하 전압 강하 (Voltage Sag) 검증"""
        T = 5
        # 전기주전자급 6A 저항성 부하
        load_kettle = {
            'p_w': np.full(T, 1320.0),
            'ih_rms': np.zeros((T, 15)),
            'ih_deg': np.zeros((T, 15)),
        }
        load_kettle['ih_rms'][:, 0] = 6.0 # 6A
        load_kettle['ih_deg'][:, 0] = 0.0
        
        result = self.engine.superimpose_parallel_loads(
            [load_kettle], apply_voltage_sag=True, v_source=220.0
        )
        
        # R_line = 0.25 Ohm -> V_sag = 6A * 0.25 Ohm = 1.5V
        # V_terminal = 220 - 1.5 = 218.5V
        self.assertTrue(np.all(result['voltage_sag'] > 1.4))
        self.assertTrue(np.all(result['vrms'] < 219.0))


if __name__ == '__main__':
    unittest.main()
