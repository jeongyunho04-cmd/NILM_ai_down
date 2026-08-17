"""
Unit Tests for NILM AI Neural Network & Loss Function
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import torch
import numpy as np
import pandas as pd

from src.models.nilm_net import MultiTaskConvBiGRUNet
from src.losses import PhysicalMultiTaskLoss
from src.inference import NILMInferenceEngine
from src.config import NUM_APPLIANCES, DEFAULT_FEATURE_DIM


class TestNILMModel(unittest.TestCase):

    def setUp(self):
        self.device = "cpu"
        self.batch_size = 4
        self.time_steps = 60
        self.in_features = DEFAULT_FEATURE_DIM
        self.model = MultiTaskConvBiGRUNet(
            in_features=self.in_features,
            num_appliances=NUM_APPLIANCES
        ).to(self.device)
        self.criterion = PhysicalMultiTaskLoss()

    def test_model_forward_shape(self):
        """모델 forward 출력 shape 검증"""
        dummy_x = torch.randn(self.batch_size, self.time_steps, self.in_features).to(self.device)
        state_logits, power_pred = self.model(dummy_x)
        
        self.assertEqual(state_logits.shape, (self.batch_size, NUM_APPLIANCES))
        self.assertEqual(power_pred.shape, (self.batch_size, NUM_APPLIANCES))
        # 전력은 ReLU로 항상 비음수(>=0)여야 함
        self.assertTrue(torch.all(power_pred >= 0.0))

    def test_loss_and_backward(self):
        """손실함수 계산 및 역전파 검증"""
        dummy_x = torch.randn(self.batch_size, self.time_steps, self.in_features).to(self.device)
        state_logits, power_pred = self.model(dummy_x)
        
        dummy_y_state = torch.randint(0, 2, (self.batch_size, NUM_APPLIANCES)).float().to(self.device)
        dummy_y_power = torch.rand(self.batch_size, NUM_APPLIANCES).to(self.device) * 100.0
        
        loss, loss_dict = self.criterion(state_logits, power_pred, dummy_y_state, dummy_y_power)
        
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("loss_cls", loss_dict)
        self.assertIn("loss_reg", loss_dict)
        self.assertIn("loss_conservation", loss_dict)
        self.assertIn("loss_state_consistency", loss_dict)
        
        # 역전파 기울기 확인
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient is None for {name}")

    def test_inference_engine(self):
        """InferenceEngine 윈도우 예측 검증"""
        engine = NILMInferenceEngine(checkpoint_path="non_existing.pt") # 초기화 테스트
        dummy_df = pd.DataFrame({
            "p_w": np.linspace(20, 100, 60),
            "irms": np.linspace(0.1, 0.5, 60),
            "vrms": [220.0]*60,
            "freq_hz": [60.0]*60,
            "thd_v": [0.01]*60,
            "phase_deg": [-30.0]*60
        })
        for h in range(1, 16):
            dummy_df[f"ih{h}"] = 0.05
            dummy_df[f"ihdeg{h}"] = -10.0
            
        pred = engine.predict_window(dummy_df)
        self.assertIn("kettle", pred["powers"])
        self.assertIn("states", pred)
        self.assertIn("powers", pred)
        self.assertIn("power_other", pred)
        self.assertIn("power_total_measured", pred)


if __name__ == '__main__':
    unittest.main()
