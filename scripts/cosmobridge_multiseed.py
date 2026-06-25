"""Multi-seed evaluation of COSMOBridge v3 to address reviewer W1.

Trains 5 gate configurations with different seeds to report mean ± std.
Both frozen paths are deterministic; only the 7 gate optimization varies.
"""

import sys, json, numpy as np, subprocess, tempfile
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from src.utils.config import load_config, get_device
from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMNS
from src.models.fusion.cosmobridge import COSMOBridge
from src.models.fusion.multimodal_pointcloud import MultimodalPointCloudModel
from src.training.metrics import compute_metrics
from scripts.train_chemprop_gbh_hybrid import ChempropGBHFusion
from scripts.train_pointcloud import PointCloudMultimodalDataset
from chemprop.utils import load_checkpoint as load_chemprop


class CachedDS(Dataset):
    def __init__(self, g, s, t, y, smi, tr):
        self.g=torch.tensor(g,dtype=torch.float32)
        self.s=torch.tensor(s,dtype=torch.float32)
        self.t=torch.tensor(t,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
        self.smi=smi; self.tr=tr
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.g[i],self.s[i],self.t[i],self.y[i],self.smi[i],self.tr[i]


def collate(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]), torch.stack([b[3] for b in batch]),
            [[b[4]] for b in batch], [b[5] for b in batch])


def identity_collate(b): return b


def extract_features(device):
    pc_dir = "data/pipeline/point_clouds"
    orig = Path("data/processed/splits")
    config = load_config("configs/default.yaml")
    pc_model = MultimodalPointCloudModel(config=config, pretrained_gnn_path=None)
    ckpt = torch.load("checkpoints/pointcloud/best_model.pt", map_location=device, weights_only=False)
    pc_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    pc_model.to(device).eval()

    data = {}
    for split in ["train","val","test"]:
        ds = PointCloudMultimodalDataset(str(orig/f"{split}.csv"), pc_dir, is_train=False)
        df = pd.read_csv(orig/f"{split}.csv")
        sf,tf,tgt=[],[],[]
        with torch.no_grad():
            for items in DataLoader(ds,batch_size=32,shuffle=False,collate_fn=identity_collate):
                sf.append(pc_model.pointnet(torch.stack([x["point_cloud"] for x in items]).to(device)).cpu().numpy())
                tf.append(torch.stack([x["features"] for x in items]).numpy())
                tgt.append(torch.stack([x["targets"] for x in items]).numpy())
        out=tempfile.mktemp(suffix=".csv")
        subprocess.run(["chemprop_fingerprint","--test_path",f"data/chemprop_tmp/{split}.csv",
                         "--features_path",f"data/chemprop_tmp/{split}_features.csv",
                         "--checkpoint_dir","checkpoints/chemprop","--fingerprint_type","MPN",
                         "--preds_path",out],capture_output=True,text=True,timeout=120)
        gf=pd.read_csv(out).select_dtypes(include=[np.number]).values.astype(np.float32)
        data[split]={"g":gf,"s":np.concatenate(sf),"t":np.concatenate(tf),"y":np.concatenate(tgt),
                      "smi":df["smiles"].tolist(),"tr":df[FEATURE_COLUMNS[:5]].values.astype(np.float32)}
    return data


def train_gates_one_seed(seed, data, fusion_model, chemprop_model, device):
    """Train COSMOBridge gates with a specific seed. Returns test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    graph_dim = data["train"]["g"].shape[1]
    from src.models.fusion.cosmobridge import COSMOBridge
    model = COSMOBridge(graph_dim=graph_dim, surface_dim=256, thermo_dim=len(FEATURE_COLUMNS),
                         fused_dim=256, rank=32, hyper_hidden=64, dropout=0.3)

    # Load frozen fusion
    cpgbh_state = fusion_model.state_dict()
    my_sd = model.state_dict()
    for cp_pre, co_pre in [("graph_proj","graph_proj"),("surface_proj","surface_proj"),
                            ("fusion","fusion"),("prediction_head","fused_head")]:
        for k,v in cpgbh_state.items():
            if k.startswith(cp_pre+"."):
                nk=k.replace(cp_pre+".",co_pre+".",1)
                if nk in my_sd and my_sd[nk].shape==v.shape: my_sd[nk]=v
    model.load_state_dict(my_sd)
    for n,p in model.named_parameters():
        if any(n.startswith(pf) for pf in ["graph_proj.","surface_proj.","fusion.","fused_head."]):
            p.requires_grad=False
    model.gate_logits.requires_grad=False
    for p in model.direct_head.parameters(): p.requires_grad=False
    # Only gates trainable
    model.gate_logits.requires_grad=True
    model.to(device)

    # Datasets
    train_ds = CachedDS(data["train"]["g"],data["train"]["s"],data["train"]["t"],data["train"]["y"],
                          data["train"]["smi"],[data["train"]["tr"][i] for i in range(len(data["train"]["smi"]))])
    val_ds = CachedDS(data["val"]["g"],data["val"]["s"],data["val"]["t"],data["val"]["y"],
                        data["val"]["smi"],[data["val"]["tr"][i] for i in range(len(data["val"]["smi"]))])
    test_ds = CachedDS(data["test"]["g"],data["test"]["s"],data["test"]["t"],data["test"]["y"],
                         data["test"]["smi"],[data["test"]["tr"][i] for i in range(len(data["test"]["smi"]))])

    train_ldr = DataLoader(train_ds,batch_size=32,shuffle=True,collate_fn=collate)
    val_ldr = DataLoader(val_ds,batch_size=32,shuffle=False,collate_fn=collate)
    test_ldr = DataLoader(test_ds,batch_size=32,shuffle=False,collate_fn=collate)

    optimizer = AdamW([model.gate_logits],lr=0.1)
    best,no_imp,best_gates=float("inf"),0,model.gate_logits.data.clone()

    for ep in range(200):
        model.train()
        for g,s,t,y,smi,tr in train_ldr:
            g,s,t,y=g.to(device),s.to(device),t.to(device),y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                pf=fusion_model(g,s,t)
                pc=chemprop_model(smi,features_batch=tr)
            alpha=torch.sigmoid(model.gate_logits)
            preds=alpha.unsqueeze(0)*pf+(1-alpha.unsqueeze(0))*pc
            loss=((preds-y)**2).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        vl,vn=0,0
        with torch.no_grad():
            for g,s,t,y,smi,tr in val_ldr:
                g,s,t,y=g.to(device),s.to(device),t.to(device),y.to(device)
                pf=fusion_model(g,s,t); pc=chemprop_model(smi,features_batch=tr)
                alpha=torch.sigmoid(model.gate_logits)
                preds=alpha.unsqueeze(0)*pf+(1-alpha.unsqueeze(0))*pc
                vl+=((preds-y)**2).mean().item(); vn+=1
        avg=vl/max(vn,1)
        if avg<best: best=avg; no_imp=0; best_gates=model.gate_logits.data.clone()
        else: no_imp+=1
        if no_imp>=40: break
    model.gate_logits.data=best_gates

    # Test
    all_p,all_t=[],[]
    model.eval()
    with torch.no_grad():
        for g,s,t,y,smi,tr in test_ldr:
            g,s,t,y=g.to(device),s.to(device),t.to(device),y.to(device)
            pf=fusion_model(g,s,t); pc=chemprop_model(smi,features_batch=tr)
            alpha=torch.sigmoid(model.gate_logits)
            preds=alpha.unsqueeze(0)*pf+(1-alpha.unsqueeze(0))*pc
            all_p.append(preds.cpu().numpy()); all_t.append(y.cpu().numpy())
    preds=np.concatenate(all_p); targets=np.concatenate(all_t)
    metrics=compute_metrics(preds,targets)
    gates=torch.sigmoid(model.gate_logits).detach().cpu().numpy()
    return metrics, gates


def main():
    device=get_device(load_config("configs/default.yaml"))
    print(f"Device: {device}")

    print("Extracting features...")
    data=extract_features(device)
    graph_dim=data["train"]["g"].shape[1]

    print("Loading frozen models...")
    fusion_model=ChempropGBHFusion(graph_dim=graph_dim,surface_dim=256,thermo_dim=len(FEATURE_COLUMNS),
                                     fused_dim=256,rank=32,hyper_hidden=64,dropout=0.3)
    fusion_model.load_state_dict(torch.load("checkpoints/chemprop_gbh_hybrid/best.pt",
                                              map_location=device,weights_only=True))
    fusion_model.to(device).eval()

    chemprop_model=load_chemprop("checkpoints/chemprop/fold_0/model_0/model.pt")
    chemprop_model.to(device).eval()

    seeds=[42,123,456,789,1024]
    all_metrics=[]
    all_gates=[]

    for seed in seeds:
        print(f"\n  Seed {seed}...")
        m,g=train_gates_one_seed(seed,data,fusion_model,chemprop_model,device)
        all_metrics.append(m)
        all_gates.append(g)
        print(f"    avg R²={m['avg_r2']:.4f}, gates=[{' '.join(f'{x:.2f}' for x in g)}]")

    # Statistics
    print(f"\n{'='*70}")
    print(f"MULTI-SEED RESULTS ({len(seeds)} seeds)")
    print(f"{'='*70}")

    print(f"\n  {'Property':<15s} {'Mean R²':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print("  "+"-"*50)
    for p in TARGET_COLUMNS:
        key=f"{p}_r2"
        vals=[m[key] for m in all_metrics]
        print(f"  {p:<15s} {np.mean(vals):8.4f} {np.std(vals):8.4f} {min(vals):8.4f} {max(vals):8.4f}")
    avgs=[m['avg_r2'] for m in all_metrics]
    print(f"  {'AVERAGE':<15s} {np.mean(avgs):8.4f} {np.std(avgs):8.4f} {min(avgs):8.4f} {max(avgs):8.4f}")

    print(f"\n  Chemprop baseline: 0.770")
    print(f"  COSMOBridge mean: {np.mean(avgs):.4f} ± {np.std(avgs):.4f}")
    print(f"  Improvement: +{np.mean(avgs)-0.770:.4f} ± {np.std(avgs):.4f}")

    results={"seeds":seeds,
             "per_seed":[{k:float(v) for k,v in m.items()} for m in all_metrics],
             "mean":{p:float(np.mean([m[f"{p}_r2"] for m in all_metrics])) for p in TARGET_COLUMNS},
             "std":{p:float(np.std([m[f"{p}_r2"] for m in all_metrics])) for p in TARGET_COLUMNS},
             "avg_mean":float(np.mean(avgs)),"avg_std":float(np.std(avgs))}
    with open("results/cosmobridge_multiseed.json","w") as f:
        json.dump(results,f,indent=2)
    print(f"\nSaved: results/cosmobridge_multiseed.json")


if __name__=="__main__":
    main()
