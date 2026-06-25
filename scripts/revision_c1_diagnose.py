"""Diagnose WHY chemprop_predict (0.770) disagrees with the manual forward call
chemprop_model(smi, features_batch=tr) (0.804) on the energy properties.

Hypothesis: the manual call returns predictions in standardized target space (or
mishandles the target scaler), while chemprop_predict applies scaler.inverse_transform.
We compare, for the test set:
  manual_raw            = chemprop_model(smi, features_batch=tr)        (what COSMOBridge uses)
  manual_inv            = target_scaler.inverse_transform(manual_raw)  (if a target scaler exists)
  cli                   = chemprop_predict output                       (official 0.770)
and report per-property R^2 of each against the SAME targets, plus max|diff| pairwise.
"""
import sys, subprocess, tempfile
from pathlib import Path
import numpy as np, pandas as pd
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import torch
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from chemprop.utils import load_checkpoint, load_scalers

CKPT = "checkpoints/chemprop/fold_0/model_0/model.pt"

def r2(yt, yp):
    return 1 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2)

df = pd.read_csv("data/chemprop_tmp/test.csv")
y = df[TARGET_COLUMNS].values.astype(np.float64)
smi = [[s] for s in df["smiles"].tolist()]
feat = pd.read_csv("data/chemprop_tmp/test_features.csv").values.astype(np.float32)

scaler, features_scaler, *_ = load_scalers(CKPT)
print("target scaler present:", scaler is not None, " features scaler present:", features_scaler is not None)
if scaler is not None:
    print("target scaler means:", np.round(np.asarray(scaler.means).ravel(), 3))
    print("target scaler stds :", np.round(np.asarray(scaler.stds).ravel(), 3))

model = load_checkpoint(CKPT); model.eval()

# manual call with raw features and with scaled features
def manual(features):
    fb = [features[i] for i in range(len(smi))]
    with torch.no_grad():
        return model(smi, features_batch=fb).cpu().numpy().astype(np.float64)

man_raw = manual(feat)
man_scaled = manual(features_scaler.transform(feat).astype(np.float32)) if features_scaler is not None else man_raw

man_raw_inv = np.asarray(scaler.inverse_transform(man_raw), dtype=np.float64) if scaler is not None else man_raw

# CLI
out = tempfile.mktemp(suffix=".csv")
subprocess.run(["chemprop_predict", "--test_path", "data/chemprop_tmp/test.csv",
                "--features_path", "data/chemprop_tmp/test_features.csv",
                "--checkpoint_dir", "checkpoints/chemprop", "--preds_path", out, "--num_workers", "0"],
               capture_output=True, text=True, timeout=600)
cli = pd.read_csv(out)[TARGET_COLUMNS].values.astype(np.float64)

print(f"\n{'prop':8s} {'man_raw':>8s} {'man_scl':>8s} {'man_inv':>8s} {'cli':>8s}")
for i, c in enumerate(TARGET_COLUMNS):
    print(f"{c:8s} {r2(y[:,i],man_raw[:,i]):8.3f} {r2(y[:,i],man_scaled[:,i]):8.3f} {r2(y[:,i],man_raw_inv[:,i]):8.3f} {r2(y[:,i],cli[:,i]):8.3f}")
print(f"{'AVG':8s} {np.mean([r2(y[:,i],man_raw[:,i]) for i in range(7)]):8.3f} "
      f"{np.mean([r2(y[:,i],man_scaled[:,i]) for i in range(7)]):8.3f} "
      f"{np.mean([r2(y[:,i],man_raw_inv[:,i]) for i in range(7)]):8.3f} "
      f"{np.mean([r2(y[:,i],cli[:,i]) for i in range(7)]):8.3f}")

print("\nmax|man_raw - cli| per prop:", np.round(np.max(np.abs(man_raw - cli), axis=0), 3))
print("max|man_inv - cli| per prop:", np.round(np.max(np.abs(man_raw_inv - cli), axis=0), 3))
print("\nINTERPRETATION: whichever column matches cli (max|diff|~0) is the correct path.")
