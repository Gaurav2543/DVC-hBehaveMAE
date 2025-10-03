"""
Predict Strain from daily-averaged frame embeddings.
Outputs:
  • PDF report   strain_prediction_<reg>_<suffix>.pdf
  • JSON metrics strain_prediction_<reg>_<suffix>.json  (incl. predictions)
  • CSV  predictions strain_prediction_<reg>_<suffix>.csv
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import csv
import seaborn as sns

# ────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────
def preprocess_embeddings(x):
    x = np.where(np.isinf(x), np.nan, x.astype(np.float64))
    x = SimpleImputer(strategy="mean").fit_transform(x)
    return StandardScaler().fit_transform(x)

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,64),    nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,num_classes)
        )
    def forward(self,x): return self.net(x)

class WideDeepMLP(nn.Module):
    def __init__(self,in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,64),     nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,num_classes)
        )
    def forward(self,x): return self.net(x)

class TabularTransformer(nn.Module):
    def __init__(self,in_dim,num_classes,d_model=128,nhead=8,nlayers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim,d_model)
        enc = nn.TransformerEncoderLayer(d_model,nhead,256)
        self.tr  = nn.TransformerEncoder(enc,nlayers)
        self.read= nn.Sequential(nn.Linear(d_model,d_model//2), nn.ReLU(),
                                 nn.Linear(d_model//2,num_classes))
    def forward(self,x):
        h=self.proj(x).unsqueeze(0)
        h=self.tr(h).mean(0)
        return self.read(h)

def train_nn(model,X_tr,y_tr,X_val,y_val,name,lr,epochs,patience):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    opt   = optim.Adam(model.parameters(),lr=lr)
    lossf = nn.CrossEntropyLoss()

    ds_tr = torch.utils.data.TensorDataset(torch.tensor(X_tr).float(),
                                           torch.tensor(y_tr).long())
    ds_vl = torch.utils.data.TensorDataset(torch.tensor(X_val).float(),
                                           torch.tensor(y_val).long())
    ld_tr = torch.utils.data.DataLoader(ds_tr,batch_size=64,shuffle=True)
    ld_vl = torch.utils.data.DataLoader(ds_vl,batch_size=64,shuffle=False)

    best,wait,hist = float("inf"),0,{"train":[],"val":[]}
    for ep in tqdm(range(epochs),desc=f"Train {name}"):
        model.train(); tr_loss=0
        for xb,yb in ld_tr:
            xb,yb=xb.to(dev),yb.to(dev)
            opt.zero_grad(); loss=lossf(model(xb),yb); loss.backward(); opt.step()
            tr_loss+=loss.item()
        hist["train"].append(tr_loss/len(ld_tr))

        model.eval(); vl_loss=0
        with torch.no_grad():
            for xb,yb in ld_vl:
                xb,yb=xb.to(dev),yb.to(dev)
                vl_loss+=lossf(model(xb),yb).item()
        vl_loss/=len(ld_vl); hist["val"].append(vl_loss)

        if vl_loss<best: best,wait=vl_loss,0; torch.save(model.state_dict(),f"best_{name}.pth")
        else: wait+=1;   
        if wait>=patience: 
            break

    model.load_state_dict(torch.load(f"best_{name}.pth"))
    return model,hist

def train_lgbm(Xtr, ytr, Xvl, yvl, num_classes):
    dtr = lgb.Dataset(Xtr, label=ytr)
    dvl = lgb.Dataset(Xvl, label=yvl)
    
    # Adjust parameters for small datasets
    n_samples = len(ytr)
    
    # Use smaller values for small datasets
    min_data_in_leaf = max(1, min(3, n_samples // 3))  # At least 1, at most 3
    num_leaves = max(2, min(8, n_samples // 2))        # At least 2, at most 8
    
    params = dict(
        objective="multiclass",
        num_class = num_classes,
        metric="multi_logloss",
        learning_rate=0.1,  # Slightly higher for small datasets
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        min_gain_to_split=0.0,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        min_data_in_bin=1,  # Add this parameter for small datasets
        verbosity=-1,       # Reduce warnings
        seed=42
    )
    
    # Reduce number of boosting rounds for small datasets
    num_boost_round = min(100, max(10, n_samples * 5))
    
    return lgb.train(
        params, 
        dtr, 
        valid_sets=[dvl], 
        num_boost_round=num_boost_round,
        callbacks=[lgb.early_stopping(10)]  # Add early stopping
    )

# ────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────
def run(args):
    # ── load data
    embobj = np.load(args.embeddings_path,allow_pickle=True).item()
    emb,fmap = embobj["embeddings"],embobj["frame_number_map"]

    lblobj = np.load(args.labels_path,allow_pickle=True).item()
    lab_arr,vocab = lblobj["label_array"],lblobj["vocabulary"]
    si,ci = vocab.index("Strain"), vocab.index("Cage")
    ai    = vocab.index("Age_Days") if "Age_Days" in vocab else None

    days,Xe,Ys,Ci,Ai=[],[],[],[],[]
    for key in sorted(fmap):
        st,en = fmap[key]
        if en<=st: continue
        days.append(key)
        Xe.append(emb[st:en].mean(0))
        Ys.append(lab_arr[si][st])
        Ci.append(lab_arr[ci][st])
        if ai is not None: Ai.append(lab_arr[ai][st])
            
    X_daily = np.vstack(Xe)
    y_daily = np.array(Ys)
    c_daily = np.array(Ci)
    a_daily = np.array(Ai) if ai is not None else None

    # Encode strain labels
    le = LabelEncoder()
    y_daily_encoded = le.fit_transform(y_daily)
    strain_classes = le.classes_
    num_classes = len(strain_classes)

    # ── train/val/test split
    gss = GroupShuffleSplit(1,test_size=args.test_frac,random_state=args.seed)
    trv_idx,te_idx = next(gss.split(X_daily,y_daily_encoded,c_daily))
    X_trv,X_te = X_daily[trv_idx],X_daily[te_idx]
    y_trv,y_te = y_daily_encoded[trv_idx],y_daily_encoded[te_idx]
    c_trv,c_te = c_daily[trv_idx],c_daily[te_idx]
    a_trv,a_te = (a_daily[trv_idx],a_daily[te_idx]) if ai is not None else (None,None)

    gss2=GroupShuffleSplit(1,test_size=0.2,random_state=args.seed)
    tr_idx,vl_idx = next(gss2.split(X_trv,y_trv,c_trv))
    X_tr,X_vl = X_trv[tr_idx],X_trv[vl_idx]
    y_tr,y_vl = y_trv[tr_idx],y_trv[vl_idx]

    # ── preprocess for NNs
    X_tr_p = preprocess_embeddings(X_tr)
    X_vl_p = preprocess_embeddings(X_vl)
    X_te_p = preprocess_embeddings(X_te)

    # ── output names
    out_prefix = f"strain_prediction_{args.regressor_type}_{args.out_suffix}"
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    pdf = PdfPages(Path(args.output_dir)/f"{out_prefix}.pdf")

    # ── strain distribution
    plt.figure(); 
    unique, counts = np.unique(y_daily, return_counts=True)
    plt.bar(unique, counts); plt.title("Strain distribution")
    plt.xlabel("Strain"); plt.ylabel("Count")
    pdf.savefig(); plt.close()

    model_type = args.regressor_type.lower()
    if model_type=="mlp":
        model,hist=train_nn(SimpleMLP(X_tr_p.shape[1], num_classes),X_tr_p,y_tr,X_vl_p,y_vl,
                            "MLP",args.mlp_lr,args.mlp_epochs,args.mlp_patience)
        plt.plot(hist["train"],label="train"); plt.plot(hist["val"],label="val")
        plt.title("MLP loss"); plt.legend(); pdf.savefig(); plt.close()
        with torch.no_grad():
            logits = model(torch.tensor(X_te_p).float().to(next(model.parameters()).device))
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    elif model_type=="wide_deep":
        model,hist=train_nn(WideDeepMLP(X_tr_p.shape[1], num_classes),X_tr_p,y_tr,X_vl_p,y_vl,
                            "WideDeep",1e-3,args.mlp_epochs,args.mlp_patience)
        plt.plot(hist["train"]); plt.plot(hist["val"]); plt.title("WideDeep loss")
        pdf.savefig(); plt.close()
        with torch.no_grad():
            logits = model(torch.tensor(X_te_p).float().to(next(model.parameters()).device))
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    elif model_type=="tabulartransformer":
        model,hist=train_nn(TabularTransformer(X_tr_p.shape[1], num_classes),X_tr_p,y_tr,X_vl_p,y_vl,
                            "Transformer",1e-3,args.mlp_epochs,args.mlp_patience)
        plt.plot(hist["train"]); plt.plot(hist["val"]); plt.title("Transformer loss")
        pdf.savefig(); plt.close()
        with torch.no_grad():
            logits = model(torch.tensor(X_te_p).float().to(next(model.parameters()).device))
            y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    elif model_type=="lgbm":
        n_comp=min(32,X_tr.shape[0],X_tr.shape[1])
        if n_comp>1:
            pca=PCA(n_components=n_comp).fit(X_tr)
            Xtr,Xvl,Xte=pca.transform(X_tr),pca.transform(X_vl),pca.transform(X_te)
        else:
            Xtr,Xvl,Xte=X_tr,X_vl,X_te
        gbm=train_lgbm(Xtr,y_tr,Xvl,y_vl,num_classes)
        y_pred_proba = gbm.predict(Xte)
        if y_pred_proba.ndim == 1:
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
        imp=gbm.feature_importance(importance_type="gain")
        plt.barh([f"PC{i+1}" for i in range(len(imp))],imp); plt.title("LGBM importance")
        pdf.savefig(); plt.close()

    else:      # logistic, svc, rf
        regs={
          "ridge":(LogisticRegression(max_iter=10000, random_state=args.seed),{"C":np.logspace(-3,3,7)}),
          "lasso":(LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000, random_state=args.seed),{"C":np.logspace(-3,3,7)}),
          "svr":  (SVC(random_state=args.seed),{"C":np.logspace(-2,2,5),"gamma":["scale","auto"]}),
          "rf":   (RandomForestClassifier(random_state=args.seed),
                   {"n_estimators":[50,100],"max_depth":[None,10,20]})
        }
        base,params=regs[model_type]
        cv_iter=list(GroupShuffleSplit(args.cv_folds,test_size=0.25,random_state=args.seed)
                     .split(X_trv,y_trv,c_trv))
        gs=GridSearchCV(base,params,scoring="accuracy",
                        cv=cv_iter,n_jobs=-1).fit(X_trv,y_trv)
        y_pred=gs.best_estimator_.predict(X_te)

    # ── metrics
    accuracy = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average='weighted')
    
    all_possible_labels = list(range(len(strain_classes)))
    class_report = classification_report(
        y_te, y_pred, 
        labels=all_possible_labels,
        target_names=strain_classes, 
        output_dict=True,
        zero_division=0  # Handle classes with no samples
    )
    
    # unique_labels = np.unique(np.concatenate([y_te, y_pred]))
    # class_report = classification_report(
    #     y_te, y_pred, 
    #     labels=unique_labels,
    #     target_names=[strain_classes[i] for i in unique_labels], 
    #     output_dict=True
    # )
    # # classification report
    # class_report = classification_report(y_te, y_pred, target_names=strain_classes, output_dict=True)
    
    # confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=strain_classes, yticklabels=strain_classes)
    plt.title(f"Confusion Matrix\nAccuracy={accuracy:.3f} F1={f1:.3f}")
    plt.xlabel("Predicted"); plt.ylabel("True")
    pdf.savefig(); plt.close()

    # per-cage accuracy
    cage_accuracy = {}
    for c in np.unique(c_te):
        mask = c_te == c
        if mask.sum() > 0:
            cage_accuracy[c] = accuracy_score(y_te[mask], y_pred[mask])
    
    plt.bar(cage_accuracy.keys(), cage_accuracy.values()); plt.title("Per-cage Accuracy")
    plt.xlabel("Cage"); plt.ylabel("Accuracy")
    pdf.savefig(); plt.close()

    # per-age accuracy (if available)
    if ai is not None:
        age_accuracy = {}
        for a in np.unique(a_te):
            mask = a_te == a
            if mask.sum() > 0:
                age_accuracy[a] = accuracy_score(y_te[mask], y_pred[mask])
        plt.bar(age_accuracy.keys(), age_accuracy.values()); plt.title("Per-age Accuracy")
        plt.xlabel("Age"); plt.ylabel("Accuracy")
        pdf.savefig(); plt.close()

    # ── save outputs
    metrics = {
        "regressor": args.regressor_type,
        "accuracy": accuracy,
        "f1_score": f1,
        "classification_report": class_report,
        "per_cage_accuracy": cage_accuracy,
        "strain_classes": strain_classes.tolist()
    }
    if ai is not None: 
        metrics["per_age_accuracy"] = age_accuracy

    # Convert predictions back to original strain labels
    y_pred_labels = le.inverse_transform(y_pred)
    y_te_labels = le.inverse_transform(y_te)

    with open(Path(args.output_dir)/f"{out_prefix}.json","w") as f:
        json.dump({**metrics,
                   "y_true": y_te_labels.tolist(),
                   "y_pred": y_pred_labels.tolist()}, f, indent=2)

    # optional small csv
    csv_path=Path(args.output_dir)/f"{out_prefix}.csv"
    with open(csv_path,"w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["seq_id","true_strain","pred_strain"])
        for key,yt,yp in zip(np.array(days)[te_idx], y_te_labels, y_pred_labels):
            wr.writerow([key,yt,yp])

    pdf.close()
    print(f"✓ PDF report  : {out_prefix}.pdf")
    print(f"✓ JSON metrics: {out_prefix}.json")
    print(f"✓ CSV preds   : {out_prefix}.csv")


# ────────────────────────────────────────────────────────
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--embeddings_path",required=True)
    ap.add_argument("--labels_path",required=True)
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--regressor_type",choices=[
        "mlp","wide_deep","tabulartransformer","lgbm",
        "ridge","lasso","svr","rf"],default="ridge")
    ap.add_argument("--mlp_epochs",type=int,default=100)
    ap.add_argument("--mlp_lr",type=float,default=1e-3)
    ap.add_argument("--mlp_patience",type=int,default=10)
    ap.add_argument("--test_frac",type=float,default=0.2)
    ap.add_argument("--cv_folds",type=int,default=5)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out_suffix",default="embeddings_strain_classifier")
    args=ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    run(args)