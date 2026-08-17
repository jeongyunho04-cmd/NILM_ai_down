"""
PyTorch Training and Evaluation Engine for NILM AI
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from .config import APPLIANCE_KEYS, APPLIANCES, NUM_APPLIANCES
from .losses import PhysicalMultiTaskLoss


def compute_metrics(
    y_state_true: np.ndarray,
    y_state_pred_probs: np.ndarray,
    y_power_true: np.ndarray,
    y_power_pred: np.ndarray,
    threshold: float = 0.5,
    gate_power_with_state: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    NILM 분류 및 전력 분해 성능 지표 계산
    - 분류: F1, Precision, Recall, Accuracy
    - 분해: MAE (W), RMSE (W), SAE (Signal Aggregate Error, %)
    """
    y_state_pred_binary = (y_state_pred_probs >= threshold).astype(np.int32)
    y_state_true_binary = (y_state_true >= threshold).astype(np.int32)
    
    # State-Gating: OFF로 예측된 기기는 전력을 0으로 강제
    if gate_power_with_state:
        effective_power_pred = y_power_pred * y_state_pred_binary
    else:
        effective_power_pred = y_power_pred
        
    eps = 1e-7
    results = {}
    
    macro_f1, macro_prec, macro_rec, macro_acc = [], [], [], []
    macro_mae, macro_rmse, macro_sae = [], [], []
    
    for i, key in enumerate(APPLIANCE_KEYS):
        # 1. Classification Metrics
        yt = y_state_true_binary[:, i]
        yp = y_state_pred_binary[:, i]
        
        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        tn = np.sum((yt == 0) & (yp == 0))
        
        acc = (tp + tn) / (len(yt) + eps)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * (prec * rec) / (prec + rec + eps)
        
        # 2. Regression Disaggregation Metrics
        pt = y_power_true[:, i]
        pp = effective_power_pred[:, i]
        
        mae = np.mean(np.abs(pp - pt))
        rmse = np.sqrt(np.mean((pp - pt)**2))
        
        # Signal Aggregate Error SAE = |sum(P_pred) - sum(P_true)| / sum(P_true)
        sae = np.abs(np.sum(pp) - np.sum(pt)) / (np.sum(pt) + eps) * 100.0
        
        results[key] = {
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "accuracy": float(acc),
            "mae_w": float(mae),
            "rmse_w": float(rmse),
            "sae_%": float(sae),
        }
        
        macro_f1.append(f1)
        macro_prec.append(prec)
        macro_rec.append(rec)
        macro_acc.append(acc)
        macro_mae.append(mae)
        macro_rmse.append(rmse)
        macro_sae.append(sae)
        
    results["overall_macro"] = {
        "f1": float(np.mean(macro_f1)),
        "precision": float(np.mean(macro_prec)),
        "recall": float(np.mean(macro_rec)),
        "accuracy": float(np.mean(macro_acc)),
        "mae_w": float(np.mean(macro_mae)),
        "rmse_w": float(np.mean(macro_rmse)),
        "sae_%": float(np.mean(macro_sae)),
    }
    
    return results


class NILMTrainer:
    """NILM AI 모델 학습, 검증 및 체크포인트 관리자"""

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[PhysicalMultiTaskLoss] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.criterion = criterion if criterion is not None else PhysicalMultiTaskLoss()
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        loss_components = {"loss_cls": 0.0, "loss_reg": 0.0, "loss_conservation": 0.0, "loss_state_consistency": 0.0}
        n_batches = 0
        
        for batch in train_loader:
            x, y_state, y_power = batch
            x = x.to(self.device)
            y_state = y_state.to(self.device)
            y_power = y_power.to(self.device)
            
            self.optimizer.zero_grad()
            state_logits, power_pred = self.model(x)
            
            loss, l_dict = self.criterion(state_logits, power_pred, y_state, y_power)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            for k in loss_components:
                loss_components[k] += l_dict.get(k, 0.0)
            n_batches += 1
            
        n = max(1, n_batches)
        res = {"loss_total": total_loss / n}
        for k in loss_components:
            res[k] = loss_components[k] / n
        return res

    def evaluate(self, val_loader: DataLoader) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        all_state_true = []
        all_state_probs = []
        all_power_true = []
        all_power_pred = []
        
        with torch.no_grad():
            for batch in val_loader:
                x, y_state, y_power = batch
                x = x.to(self.device)
                y_state = y_state.to(self.device)
                y_power = y_power.to(self.device)
                
                state_logits, power_pred = self.model(x)
                loss, _ = self.criterion(state_logits, power_pred, y_state, y_power)
                
                total_loss += loss.item()
                n_batches += 1
                
                probs = torch.sigmoid(state_logits).cpu().numpy()
                all_state_probs.append(probs)
                all_state_true.append(y_state.cpu().numpy())
                all_power_pred.append(power_pred.cpu().numpy())
                all_power_true.append(y_power.cpu().numpy())
                
        n = max(1, n_batches)
        loss_avg = {"val_loss": total_loss / n}
        
        y_st_true = np.concatenate(all_state_true, axis=0)
        y_st_prob = np.concatenate(all_state_probs, axis=0)
        y_pw_true = np.concatenate(all_power_true, axis=0)
        y_pw_pred = np.concatenate(all_power_pred, axis=0)
        
        metrics = compute_metrics(y_st_true, y_st_prob, y_pw_true, y_pw_pred)
        return loss_avg, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        checkpoint_path: str = "checkpoints/best_nilm_model.pt",
        early_stopping_patience: int = 7
    ) -> Dict[str, List[float]]:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        
        best_f1 = -1.0
        best_mae = float("inf")
        patience_cnt = 0
        
        history = {
            "train_loss": [], "val_loss": [], "val_f1": [], "val_mae": []
        }
        
        print(f"\n[+] NILM AI 모델 학습 시작 (Device: {self.device}, Epochs: {epochs})")
        print("-" * 80)
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_res = self.train_epoch(train_loader)
            val_loss, metrics = self.evaluate(val_loader)
            scheduler.step()
            
            elapsed = time.time() - t0
            ov = metrics["overall_macro"]
            
            history["train_loss"].append(train_res["loss_total"])
            history["val_loss"].append(val_loss["val_loss"])
            history["val_f1"].append(ov["f1"])
            history["val_mae"].append(ov["mae_w"])
            
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) "
                f"Train Loss: {train_res['loss_total']:.4f} | Val Loss: {val_loss['val_loss']:.4f} | "
                f"Val F1: {ov['f1']*100:.2f}% | Val MAE: {ov['mae_w']:.2f}W (SAE: {ov['sae_%']:.1f}%)"
            )
            
            # Best Model Criterion: 높은 F1 및 낮은 MAE 복합 점수
            # composite_score: F1(0~1) + (100 / (100 + MAE))
            comp_score = ov["f1"] + (50.0 / (50.0 + ov["mae_w"]))
            
            if comp_score > best_f1:
                best_f1 = comp_score
                best_mae = ov["mae_w"]
                patience_cnt = 0
                
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_f1": ov["f1"],
                    "val_mae": best_mae,
                    "metrics": metrics
                }, checkpoint_path)
                print(f"  --> [*] Best Checkpoint 저장됨 (F1: {ov['f1']*100:.2f}%, MAE: {best_mae:.2f}W, Score: {comp_score:.3f})")
            else:
                patience_cnt += 1
                if patience_cnt >= early_stopping_patience:
                    print(f"\n[!] Early stopping 발동 (Epoch {epoch})")
                    break
                    
        print("-" * 80)
        print(f"[+] 학습 완료! 최고 성능 -> F1: {best_f1*100:.2f}%, MAE: {best_mae:.2f}W")
        return history
