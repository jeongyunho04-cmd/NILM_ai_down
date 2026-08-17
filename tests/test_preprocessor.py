"""
Unit Tests for NILM Preprocessing Pipeline
"""
import unittest
import numpy as np
import pandas as pd
import os

from src.data_cleaner import clean_dataframe
from src.feature_engineering import extract_features, get_feature_column_names
from src.event_labeler import label_appliance_state
from src.synthetic_generator import SyntheticDatasetGenerator
from src.dataset import NILMDataset
from src.config import NUM_HARMONICS, APPLIANCE_KEYS


class TestNILMPreprocessor(unittest.TestCase):

    def test_data_cleaner(self):
        """데이터 클리너: 음수 클리핑 및 PLL 언락 보간 테스트"""
        df = pd.DataFrame({
            'p_w': [-10.0, 50.0, 50.0, 50.0, -5.0],
            'irms': [-0.1, 0.25, 0.25, 0.25, 0.0],
            'pll_locked': [1, 1, 0, 1, 1], # 2번 인덱스가 PLL 언락
            'freq_hz': [60.0, 60.0, 0.0, 60.0, 60.0],
            'vrms': [220.0, 220.0, 0.0, 220.0, 220.0],
            'ih1': [0.0, 0.25, 0.0, 0.25, 0.0]
        })
        
        cleaned = clean_dataframe(df, interpolate_pll_unlock=True)
        
        # 음수 클리핑 확인
        self.assertGreaterEqual(cleaned['p_w'].min(), 0.0)
        self.assertGreaterEqual(cleaned['irms'].min(), 0.0)
        # PLL 언락 구간 보간 확인 (2번 인덱스의 vrms가 0이 아니라 220으로 보간되어야 함)
        self.assertAlmostEqual(cleaned['vrms'].iloc[2], 220.0, delta=1.0)
        self.assertAlmostEqual(cleaned['freq_hz'].iloc[2], 60.0, delta=0.5)

    def test_feature_engineering(self):
        """피처 엔지니어링: S, Q, PF, THD_I 및 직교 페이저 추출 테스트"""
        df = pd.DataFrame({
            'p_w': [220.0],
            'vrms': [220.0],
            'irms': [1.4142], # S = 220 * 1.4142 = 311.124 VA
            'phase_deg': [-45.0], # 45도 지상
            'freq_hz': [60.0],
            'ih1': [1.0]
        })
        for h in range(2, NUM_HARMONICS + 1):
            df[f'ih{h}'] = 0.0
            df[f'ihdeg{h}'] = 0.0
        df['ihdeg1'] = -45.0
        df['ih3'] = 0.5 # 3차 고조파 0.5A
        df['ihdeg3'] = 90.0
        
        feat_df = extract_features(df)
        
        # S = V * I = 220 * 1.4142 = 311.124
        self.assertAlmostEqual(feat_df['s_va'].iloc[0], 311.124, delta=0.5)
        # PF = P / S = 220 / 311.124 = 0.707
        self.assertAlmostEqual(feat_df['power_factor'].iloc[0], 0.707, delta=0.01)
        # THD_I = sqrt(0.5^2) / 1.0 = 0.50
        self.assertAlmostEqual(feat_df['thd_i'].iloc[0], 0.50, delta=0.01)
        # 3차 고조파 직교 성분: Re = 0.5 * cos(90) = 0, Im = 0.5 * sin(90) = 0.5
        self.assertAlmostEqual(feat_df['ih_re_3'].iloc[0], 0.0, delta=1e-4)
        self.assertAlmostEqual(feat_df['ih_im_3'].iloc[0], 0.5, delta=1e-4)
        # 직교 위상 성분: cos(90) = 0, sin(90) = 1
        self.assertAlmostEqual(feat_df['ih_cos_3'].iloc[0], 0.0, delta=1e-4)
        self.assertAlmostEqual(feat_df['ih_sin_3'].iloc[0], 1.0, delta=1e-4)

    def test_event_labeler(self):
        """이벤트 라벨러: Schmitt Trigger 및 히스테리시스 필터링 테스트"""
        # 10W 미만 OFF, 15W 이상 ON
        p_seq = [0.0]*10 + [25.0]*30 + [12.0]*5 + [25.0]*20 + [0.0]*10
        df = pd.DataFrame({'p_w': p_seq})
        
        labeled = label_appliance_state(df, appliance_key="fan", on_thresh=15.0, off_thresh=8.0)
        
        # 12W 구간(인덱스 40~44)에서도 직전 상태가 ON이었으므로 계속 1이어야 함 (히스테리시스)
        self.assertEqual(labeled['is_on'].iloc[42], 1)
        # 초기 및 끝단은 0이어야 함
        self.assertEqual(labeled['is_on'].iloc[0], 0)
        self.assertEqual(labeled['is_on'].iloc[-1], 0)

    def test_synthetic_generator_and_dataset(self):
        """합성 생성기 및 NILM Dataset 종합 연동 테스트"""
        generator = SyntheticDatasetGenerator()
        
        # 10초 분량 시나리오 생성 (600스텝)
        agg_df, gt_df = generator.generate_scenario(duration_seconds=10.0, seed=42)
        
        self.assertEqual(len(agg_df), 600)
        self.assertEqual(len(gt_df), 600)
        
        # Ground truth 컬럼 확인
        for k in APPLIANCE_KEYS:
            self.assertIn(f"state_{k}", gt_df.columns)
            self.assertIn(f"power_{k}", gt_df.columns)
            
        # NILM Dataset 윈도우 생성 (Window=60, Stride=30 -> 19개 샘플)
        dataset = NILMDataset(agg_df, gt_df, window_size=60, step_stride=30)
        self.assertGreater(len(dataset), 0)
        
        x_sample, y_state, y_power = dataset[0]
        # x_sample shape: (60, num_features)
        self.assertEqual(x_sample.shape[0], 60)
        self.assertEqual(len(y_state), len(APPLIANCE_KEYS))
        self.assertEqual(len(y_power), len(APPLIANCE_KEYS))


if __name__ == '__main__':
    unittest.main()
