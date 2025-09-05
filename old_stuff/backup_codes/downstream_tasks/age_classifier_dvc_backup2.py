"""
Predict Age_Days from daily-averaged frame embeddings.
Outputs:
  • PDF report   age_prediction_<reg>_<suffix>.pdf
  • JSON metrics age_prediction_<reg>_<suffix>.json  (incl. predictions)
  • CSV  predictions age_prediction_<reg>_<suffix>.csv
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import csv

# ────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────
def preprocess_embeddings(x):
    x = np.where(np.isinf(x), np.nan, x.astype(np.float64))
    x = SimpleImputer(strategy="mean").fit_transform(x)
    return StandardScaler().fit_transform(x)

class SimpleMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,64),    nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,1)
        )
    def forward(self,x): return self.net(x)

class WideDeepMLP(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,64),     nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,1)
        )
    def forward(self,x): return self.net(x)

class TabularTransformer(nn.Module):
    def __init__(self,in_dim,d_model=128,nhead=8,nlayers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim,d_model)
        enc = nn.TransformerEncoderLayer(d_model,nhead,256)
        self.tr  = nn.TransformerEncoder(enc,nlayers)
        self.read= nn.Sequential(nn.Linear(d_model,d_model//2), nn.ReLU(),
                                 nn.Linear(d_model//2,1))
    def forward(self,x):
        h=self.proj(x).unsqueeze(0)
        h=self.tr(h).mean(0)
        return self.read(h)

def train_nn(model,X_tr,y_tr,X_val,y_val,name,lr,epochs,patience):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    opt   = optim.Adam(model.parameters(),lr=lr)
    lossf = nn.MSELoss()

    ds_tr = torch.utils.data.TensorDataset(torch.tensor(X_tr).float(),
                                           torch.tensor(y_tr).float().view(-1,1))
    ds_vl = torch.utils.data.TensorDataset(torch.tensor(X_val).float(),
                                           torch.tensor(y_val).float().view(-1,1))
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

# def train_lgbm(Xtr,ytr,Xvl,yvl):
#     dtr = lgb.Dataset(Xtr,label=ytr)
#     dvl = lgb.Dataset(Xvl,label=yvl)
#     params = dict(objective="regression",metric="rmse",
#                   learning_rate=0.05,num_leaves=15,
#                   min_data_in_leaf=5,min_gain_to_split=0.0,
#                   feature_fraction=1.0,bagging_fraction=1.0,seed=42)
#     return lgb.train(params,dtr,valid_sets=[dvl],num_boost_round=300)

def train_lgbm(Xtr, ytr, Xvl, yvl):
    dtr = lgb.Dataset(Xtr, label=ytr)
    dvl = lgb.Dataset(Xvl, label=yvl)
    
    # Adjust parameters for small datasets
    n_samples = len(ytr)
    
    # Use smaller values for small datasets
    min_data_in_leaf = max(1, min(3, n_samples // 3))  # At least 1, at most 3
    num_leaves = max(2, min(7, n_samples // 2))        # At least 2, at most 7
    
    params = dict(
        objective="regression",
        metric="rmse",
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
    ai,ci = vocab.index("Age_Days"), vocab.index("Cage")
    si    = vocab.index("Strain") if "Strain" in vocab else None

    days,Xe,Ya,Ci,Si=[],[],[],[],[]
    for key in sorted(fmap):
        st,en = fmap[key]
        # if en<=st: continue
        # days.append(key)
        # Xe.append(emb[st:en].mean(0))
        # Ya.append(lab_arr[ai][st])
        # Ci.append(lab_arr[ci][st])
        # if si is not None: Si.append(lab_arr[si][st])
        # skip sequences that have no frames
        if en <= st:
            continue
        # build daily embedding
        emb_day = emb[st:en].mean(0)
        # protect against all-NaN rows
        if np.isnan(emb_day).all():
            continue
        days.append(key)
        Xe.append(emb_day)
        Ya.append(lab_arr[ai][st])
        Ci.append(lab_arr[ci][st])
        if si is not None:
            Si.append(lab_arr[si][st])
            
    X_daily = np.vstack(Xe)
    y_daily = np.array(Ya)
    c_daily = np.array(Ci)
    s_daily = np.array(Si) if si is not None else None

    # ── train/val/test split
    gss = GroupShuffleSplit(1,test_size=args.test_frac,random_state=args.seed)
    trv_idx,te_idx = next(gss.split(X_daily,y_daily,c_daily))
    X_trv,X_te = X_daily[trv_idx],X_daily[te_idx]
    y_trv,y_te = y_daily[trv_idx],y_daily[te_idx]
    c_trv,c_te = c_daily[trv_idx],c_daily[te_idx]
    s_trv,s_te = (s_daily[trv_idx],s_daily[te_idx]) if si is not None else (None,None)

    gss2=GroupShuffleSplit(1,test_size=0.2,random_state=args.seed)
    tr_idx,vl_idx = next(gss2.split(X_trv,y_trv,c_trv))
    X_tr,X_vl = X_trv[tr_idx],X_trv[vl_idx]
    y_tr,y_vl = y_trv[tr_idx],y_trv[vl_idx]

    # ── preprocess for NNs
    X_tr_p = preprocess_embeddings(X_tr)
    X_vl_p = preprocess_embeddings(X_vl)
    X_te_p = preprocess_embeddings(X_te)

    # ── output names
    out_prefix = f"age_prediction_{args.regressor_type}_{args.out_suffix}"
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    pdf = PdfPages(Path(args.output_dir)/f"{out_prefix}.pdf")

    # ── age distribution
    plt.figure(); plt.hist(y_daily,bins=30); plt.title("Age distribution")
    pdf.savefig(); plt.close()

    model_type = args.regressor_type.lower()
    if model_type=="mlp":
        model,hist=train_nn(SimpleMLP(X_tr_p.shape[1]),X_tr_p,y_tr,X_vl_p,y_vl,
                            "MLP",args.mlp_lr,args.mlp_epochs,args.mlp_patience)
        plt.plot(hist["train"],label="train"); plt.plot(hist["val"],label="val")
        plt.title("MLP loss"); plt.legend(); pdf.savefig(); plt.close()
        y_pred = model(torch.tensor(X_te_p).float().to(next(model.parameters()).device)).detach().cpu().numpy().flatten()

    elif model_type=="wide_deep":
        model,hist=train_nn(WideDeepMLP(X_tr_p.shape[1]),X_tr_p,y_tr,X_vl_p,y_vl,
                            "WideDeep",1e-3,args.mlp_epochs,args.mlp_patience)
        plt.plot(hist["train"]); plt.plot(hist["val"]); plt.title("WideDeep loss")
        pdf.savefig(); plt.close()
        y_pred = model(torch.tensor(X_te_p).float().to(next(model.parameters()).device)).detach().cpu().numpy().flatten()

    elif model_type=="tabulartransformer":
        model,hist=train_nn(TabularTransformer(X_tr_p.shape[1]),X_tr_p,y_tr,X_vl_p,y_vl,
                            "Transformer",1e-3,args.mlp_epochs,args.mlp_patience)
        plt.plot(hist["train"]); plt.plot(hist["val"]); plt.title("Transformer loss")
        pdf.savefig(); plt.close()
        y_pred = model(torch.tensor(X_te_p).float().to(next(model.parameters()).device)).detach().cpu().numpy().flatten()

    elif model_type=="lgbm":
        n_comp=min(32,X_tr.shape[0],X_tr.shape[1])
        if n_comp>1:
            pca=PCA(n_components=n_comp).fit(X_tr)
            Xtr,Xvl,Xte=pca.transform(X_tr),pca.transform(X_vl),pca.transform(X_te)
        else:
            Xtr,Xvl,Xte=X_tr,X_vl,X_te
        gbm=train_lgbm(Xtr,y_tr,Xvl,y_vl)
        y_pred=gbm.predict(Xte)
        imp=gbm.feature_importance(importance_type="gain")
        plt.barh([f"PC{i+1}" for i in range(len(imp))],imp); plt.title("LGBM importance")
        pdf.savefig(); plt.close()

    else:      # ridge, lasso, svr, rf
        regs={
          "ridge":(Ridge(),{"alpha":np.logspace(-3,3,7)}),
          "lasso":(Lasso(max_iter=10000),{"alpha":np.logspace(-3,3,7)}),
          "svr":  (SVR(),{"C":np.logspace(-2,2,5),"gamma":["scale","auto"]}),
          "rf":   (RandomForestRegressor(random_state=args.seed),
                   {"n_estimators":[50,100],"max_depth":[None,10,20]})
        }
        base,params=regs[model_type]
        # cv_iter=list(GroupShuffleSplit(args.cv_folds,test_size=0.25,random_state=args.seed)
        #              .split(X_trv,y_trv,c_trv))
        # gs=GridSearchCV(base,params,scoring="neg_root_mean_squared_error",
        #                 cv=cv_iter,n_jobs=-1).fit(X_trv,y_trv)
        # y_pred=gs.best_estimator_.predict(X_te)
        # --- PREPROCESS the whole train-val set before CV -----------------
        X_trv_p = preprocess_embeddings(X_trv)
        X_te_p  = preprocess_embeddings(X_te)          # keep pair with X_trv_p
  
        cv_iter = list(GroupShuffleSplit(args.cv_folds,test_size=0.25,random_state=args.seed).split(X_trv_p, y_trv, c_trv))
        gs = GridSearchCV(base, params,
                        scoring="neg_root_mean_squared_error",
                        cv=cv_iter, n_jobs=-1, error_score="raise")
        gs.fit(X_trv_p, y_trv)
   
        best = gs.best_estimator_
        y_pred = best.predict(X_te_p)

    # ── metrics
    rmse=np.sqrt(mean_squared_error(y_te,y_pred))
    mae =      mean_absolute_error(y_te,y_pred)
    r2  =             r2_score(y_te,y_pred)

    # scatter
    plt.scatter(y_te,y_pred,s=10,alpha=0.5)
    lim=[min(y_te.min(),y_pred.min()),max(y_te.max(),y_pred.max())]
    plt.plot(lim,lim,"k--"); plt.title(f"True vs Pred\nRMSE={rmse:.2f} MAE={mae:.2f} R²={r2:.2f}")
    pdf.savefig(); plt.close()

    # residuals
    resid=y_pred-y_te
    plt.hist(resid,bins=30); plt.title("Residuals"); pdf.savefig(); plt.close()

    # per-cage
    cage_rmse={c:np.sqrt(mean_squared_error(y_te[c_te==c],y_pred[c_te==c])) for c in np.unique(c_te)}
    plt.bar(cage_rmse.keys(),cage_rmse.values()); plt.title("Per-cage RMSE")
    plt.xticks(rotation=45, ha='right')
    pdf.savefig(); plt.close()

    # per-strain
    if si is not None:
        strain_rmse={s:np.sqrt(mean_squared_error(y_te[s_te==s],y_pred[s_te==s])) for s in np.unique(s_te)}
        plt.bar(strain_rmse.keys(),strain_rmse.values()); plt.title("Per-strain RMSE")
        plt.xticks(rotation=45, ha='right')
        pdf.savefig(); plt.close()

    # ── save outputs
    metrics={
        "regressor":args.regressor_type,
        "RMSE":rmse,"MAE":mae,"R2":r2,
        "per_cage_rmse":cage_rmse
    }
    if si is not None: metrics["per_strain_rmse"]=strain_rmse

    with open(Path(args.output_dir)/f"{out_prefix}.json","w") as f:
        json.dump({**metrics,
                   "y_true":y_te.tolist(),
                   "y_pred":y_pred.tolist()},f,indent=2)

    # optional small csv
    csv_path=Path(args.output_dir)/f"{out_prefix}.csv"
    with open(csv_path,"w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["seq_id","true_age","pred_age"])
        for key,yt,yp in zip(np.array(days)[te_idx],y_te,y_pred):
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
    ap.add_argument("--out_suffix",default="embeddings_age_classifier")
    args=ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    run(args)