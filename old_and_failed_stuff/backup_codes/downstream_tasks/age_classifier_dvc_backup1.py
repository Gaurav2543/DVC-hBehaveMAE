"""
Aggregates frame embeddings into daily averages, fits a variety of regressors to predict Age_Days,
and produces an extensive PDF report with:
    • Train/Val loss (for NN models)
    • True vs Predicted scatter
    • Distribution of true ages
    • Residual histograms & residual vs age
    • Per-Cage and Per-Strain RMSE bar charts
    • Feature importance (for LightGBM)
Usage:
    python age_classifier.py \
      --embeddings_path path/to/emb_combined.npy \
      --labels_path     path/to/arrays_labels.npy \
      --output_dir      results/ \
      --regressor_type  lgbm
"""
import os, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.impute      import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm         import SVR
from sklearn.ensemble    import RandomForestRegressor
from sklearn.metrics     import mean_squared_error, r2_score

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions & models
# ────────────────────────────────────────────────────────────────────────────────
def preprocess_embeddings(X):
    X = np.where(np.isinf(X), np.nan, X.astype(np.float64))
    X = SimpleImputer(strategy="mean").fit_transform(X)
    return StandardScaler().fit_transform(X)

class SimpleMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

class WideDeepMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,1)
        )
    def forward(self, x): return self.net(x)

class TabularTransformer(nn.Module):
    def __init__(self, in_dim, d_model=128, nhead=8, nlayers=2):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256)
        self.trans  = nn.TransformerEncoder(enc_layer, nlayers)
        self.read   = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2,1))
    def forward(self, x):
        h = self.proj(x).unsqueeze(0)
        h = self.trans(h).mean(dim=0)
        return self.read(h)

def train_nn(model, X_tr, y_tr, X_val, y_val, name, lr=1e-3, epochs=100, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    # data loaders
    tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr).float(),
                                           torch.from_numpy(y_tr).float().view(-1,1))
    vl_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(),
                                           torch.from_numpy(y_val).float().view(-1,1))
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=64, shuffle=True)
    vl_ld = torch.utils.data.DataLoader(vl_ds, batch_size=64, shuffle=False)

    best, wait = float("inf"), 0
    history = {"train_loss": [], "val_loss": []}
    for ep in tqdm(range(epochs), desc=f"Train {name}"):
        # train
        model.train()
        Ltr = 0
        for xb,yb in tr_ld:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            Ltr += loss.item()
        history["train_loss"].append(Ltr/len(tr_ld))
        # val
        model.eval()
        Lv=0
        with torch.no_grad():
            for xb,yb in vl_ld:
                xb,yb = xb.to(device), yb.to(device)
                Lv += loss_fn(model(xb), yb).item()
        Lv /= len(vl_ld)
        history["val_loss"].append(Lv)
        # early stop
        if Lv<best:
            best, wait = Lv, 0
            torch.save(model.state_dict(), f"best_{name}.pth")
        else:
            wait+=1
            if wait>=patience: break
    model.load_state_dict(torch.load(f"best_{name}.pth"))
    return model, history

def train_lgbm(X_tr, y_tr, X_val, y_val):
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dvl = lgb.Dataset(X_val, label=y_val)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 15,           # fewer leaves for tiny data
        "min_data_in_leaf": 5,      # allow small leaves
        "min_gain_to_split": 0.0,   # disable gain threshold
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "seed": 42,
    }
    gbm = lgb.train(params, dtr, valid_sets=[dvl], num_boost_round=300)
    return gbm

# ────────────────────────────────────────────────────────────────────────────────
# Main training / evaluation & plotting
# ────────────────────────────────────────────────────────────────────────────────
def run_age_prediction(args):
    # load embeddings + frame_map
    dd = np.load(args.embeddings_path, allow_pickle=True).item()
    emb, fmap = dd["embeddings"], dd["frame_number_map"]
    # load labels
    ld = np.load(args.labels_path, allow_pickle=True).item()
    # print number of cages and strains
    # print(f"Loaded {len(ld['cage'])} cages, {len(ld['strain'])} strains")
    # cage_array, strain_array = ld["cage_array"], ld["strain_array"]
    lab_arr, vocab = ld["label_array"], ld["vocabulary"]

    # find indices
    ai = vocab.index("Age_Days")
    ci = vocab.index("Cage")
    si = vocab.index("Strain") if "Strain" in vocab else None

    # aggregate per day
    days, Xe, Ya, Ci, Si = [], [], [], [], []
    for key in sorted(fmap):
        st, en = fmap[key]
        if en<=st: continue
        days.append(key)
        Xe.append( emb[st:en].mean(0) )
        # Ya.append( lab_arr[ai, st] )
        Ya.append( lab_arr[ai][st] )
        # Ci.append( lab_arr[ci, st] )
        Ci.append( lab_arr[ci][st] )
        # if si is not None: Si.append(lab_arr[si, st])
        if si is not None: Si.append(lab_arr[si][st])
    X_daily = np.vstack(Xe)
    y_daily = np.array(Ya)
    c_daily = np.array(Ci)
    s_daily = np.array(Si) if si is not None else None

    # splits
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    trv_idx, te_idx = next(gss.split(X_daily, y_daily, c_daily))
    X_trv, X_te = X_daily[trv_idx], X_daily[te_idx]
    y_trv, y_te = y_daily[trv_idx],  y_daily[te_idx]
    c_trv, c_te = c_daily[trv_idx],  c_daily[te_idx]
    s_trv, s_te = (s_daily[trv_idx], s_daily[te_idx]) if si is not None else (None,None)

    # further split trv→train/val for NN
    gss2=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=args.seed)
    tr_idx, vl_idx = next(gss2.split(X_trv,y_trv,c_trv))
    X_tr, X_vl = X_trv[tr_idx], X_trv[vl_idx]
    y_tr, y_vl = y_trv[tr_idx], y_trv[vl_idx]

    # preprocess
    X_tr_p = preprocess_embeddings(X_tr)
    X_vl_p = preprocess_embeddings(X_vl)
    X_te_p = preprocess_embeddings(X_te)

    # prepare PDF
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_prefix = f"age_prediction_{args.regressor_type}_{args.out_suffix}"
    pdf = PdfPages(Path(args.output_dir) / f"{out_prefix}.pdf")
    # pdf = PdfPages(Path(args.output_dir)/f"{args.out_suffix}_plots.pdf")

    # 1) PRESS WHOLE-DATA AGE DISTRIBUTION
    plt.figure()
    plt.hist(y_daily, bins=30, color="gray", edgecolor="k")
    plt.title("Overall Age Distribution")
    plt.xlabel("Age_Days"); plt.ylabel("Count")
    pdf.savefig(); plt.close()

    # Train specified regressor
    model_type = args.regressor_type.lower()
    if model_type=="mlp":
        model, hist = train_nn(SimpleMLP(X_tr_p.shape[1]), X_tr_p, y_tr, X_vl_p, y_vl, "MLP",
                                lr=args.mlp_lr, epochs=args.mlp_epochs, patience=args.mlp_patience)
        # plot loss
        fig,ax=plt.subplots()
        ax.plot(hist["train_loss"], label="Train"); ax.plot(hist["val_loss"], label="Val")
        ax.set_title("MLP Train/Val Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.legend()
        pdf.savefig(fig); plt.close(fig)

        y_pred = model(torch.from_numpy(X_te_p).float().to(next(model.parameters()).device)).cpu().detach().numpy().flatten()

    elif model_type=="wide_deep":
        model, hist = train_nn(WideDeepMLP(X_tr_p.shape[1]), X_tr_p, y_tr, X_vl_p, y_vl, "WideDeep",
                               lr=1e-3, epochs=args.mlp_epochs, patience=args.mlp_patience)
        fig,ax=plt.subplots()
        ax.plot(hist["train_loss"], label="Train"); ax.plot(hist["val_loss"], label="Val")
        ax.set_title("WideDeep Train/Val Loss"); ax.legend()
        pdf.savefig(fig); plt.close(fig)
        y_pred = model(torch.from_numpy(X_te_p).float().to(next(model.parameters()).device)).cpu().detach().numpy().flatten()

    elif model_type=="tabulartransformer":
        model, hist = train_nn(TabularTransformer(X_tr_p.shape[1]), X_tr_p, y_tr, X_vl_p, y_vl, "Transformer",
                               lr=1e-3, epochs=args.mlp_epochs, patience=args.mlp_patience)
        fig,ax=plt.subplots()
        ax.plot(hist["train_loss"], label="Train"); ax.plot(hist["val_loss"], label="Val")
        ax.set_title("Transformer Train/Val Loss"); ax.legend()
        pdf.savefig(fig); plt.close(fig)
        y_pred = model(torch.from_numpy(X_te_p).float().to(next(model.parameters()).device)).cpu().detach().numpy().flatten()

    elif model_type=="lgbm":
        n_train, n_feat = X_tr.shape
        # cap at min(32, n_feat, n_train)
        n_comp = min(32, n_feat, n_train)
        if n_comp <= 1:
            # too small to PCA—just pass raw
            Xl_tr, Xl_vl, Xl_te = X_tr, X_vl, X_te
        else:
            pca = PCA(n_components=n_comp).fit(X_tr)
            Xl_tr = pca.transform(X_tr)
            Xl_vl = pca.transform(X_vl)
            Xl_te = pca.transform(X_te)

        # train LightGBM on reduced features
        gbm = train_lgbm(Xl_tr, y_tr, Xl_vl, y_vl)
        y_pred = gbm.predict(Xl_te)

        # plot feature importance
        imp = gbm.feature_importance(importance_type="gain")
        feat = [f"PC{i+1}" for i in range(len(imp))]
        fig, ax = plt.subplots(figsize=(6,6))
        ax.barh(feat, imp)
        ax.set_title("LightGBM Feature Importance")
        pdf.savefig(fig)
        plt.close(fig)

    else:
        # linear / tree models via sklearn + CV
        regs = {
            "ridge":   (Ridge(),  {"alpha": np.logspace(-3,3,7)}),
            "lasso":   (Lasso(max_iter=10_000), {"alpha": np.logspace(-3,3,7)}),
            "svr":     (SVR(), {"C": np.logspace(-2,2,5),"gamma":["scale","auto"]}),
            "rf":      (RandomForestRegressor(random_state=args.seed), {"n_estimators":[50,100],"max_depth":[None,10,20]})
        }
        base, params = regs[model_type]
        gsscv = GroupShuffleSplit(n_splits=args.cv_folds, test_size=0.25, random_state=args.seed)
        cv_iter = list(gsscv.split(X_trv, y_trv, c_trv))
        gs = GridSearchCV(base, params, scoring="neg_root_mean_squared_error", cv=cv_iter, n_jobs=-1)
        gs.fit(X_trv, y_trv)
        best = gs.best_estimator_
        y_pred = best.predict(X_te)

        # hyperparam plot if single param
        if len(params)==1:
            pnm = next(iter(params))
            scores = -gs.cv_results_["mean_test_score"]
            fig,ax=plt.subplots()
            ax.semilogx(params[pnm], scores, marker="o")
            ax.set_title(f"{model_type} CV {pnm} vs RMSE")
            pdf.savefig(fig); plt.close(fig)

    # common metrics
    mse = mean_squared_error(y_te, y_pred)
    rmse = np.sqrt(mse)
    r2  = r2_score(y_te, y_pred)

    # true vs pred
    fig,ax=plt.subplots(figsize=(6,6))
    ax.scatter(y_te, y_pred, alpha=0.5, s=10)
    mn,mx = min(y_te.min(),y_pred.min()), max(y_te.max(),y_pred.max())
    ax.plot([mn,mx],[mn,mx],"k--")
    ax.set_title(f"True vs Pred (RMSE={rmse:.2f}, R2={r2:.2f})")
    ax.set_xlabel("True"); ax.set_ylabel("Pred")
    pdf.savefig(fig); plt.close(fig)

    # residual histogram
    resid = y_pred - y_te
    fig,ax=plt.subplots()
    ax.hist(resid, bins=30, color="tomato", edgecolor="k")
    ax.set_title("Residuals Distribution")
    pdf.savefig(fig); plt.close(fig)

    # residual vs true
    fig,ax=plt.subplots()
    ax.scatter(y_te, resid, alpha=0.5, s=10)
    ax.axhline(0,color="k",ls="--")
    ax.set_title("Residual vs True Age"); ax.set_xlabel("True"); ax.set_ylabel("Error")
    pdf.savefig(fig); plt.close(fig)

    # per-Cage RMSE
    cages = np.unique(c_te)
    cage_rmse = {c: np.sqrt(mean_squared_error(y_te[c_te==c], y_pred[c_te==c])) for c in cages}
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(list(cage_rmse.keys()), list(cage_rmse.values()), color="skyblue")
    ax.set_title("Per-Cage RMSE"); ax.set_xlabel("Cage"); ax.set_ylabel("RMSE")
    plt.xticks(rotation=45, ha="right")
    pdf.savefig(fig); plt.close(fig)

    # per-Strain RMSE
    if si is not None:
        strains = np.unique(s_te)
        str_rmse = {s: np.sqrt(mean_squared_error(y_te[s_te==s], y_pred[s_te==s])) for s in strains}
        fig,ax=plt.subplots(figsize=(8,4))
        ax.bar(list(str_rmse.keys()), list(str_rmse.values()), color="lightgreen")
        ax.set_title("Per-Strain RMSE"); ax.set_xlabel("Strain"); ax.set_ylabel("RMSE")
        plt.xticks(rotation=90, ha="right")
        pdf.savefig(fig); plt.close(fig)

    # save results
    results = {
        "regressor": args.regressor_type,
        "RMSE": rmse, "R2": r2,
        "per_cage_rmse": cage_rmse,
        **({"per_strain_rmse": str_rmse} if si is not None else {})
    }
    # with open(Path(args.output_dir)/f"{args.out_suffix}_results.json","w") as f:
    with open(Path(args.output_dir) / f"{out_prefix}.json", "w") as f:
        json.dump(results, f, indent=2)

    pdf.close()
    print(f"✓ report  → {out_prefix}.pdf")
    print(f"✓ metrics → {out_prefix}.json")

# ────────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_path", required=True)
    p.add_argument("--labels_path",     required=True)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--regressor_type",  choices=[
        "mlp","wide_deep","tabulartransformer","lgbm","ridge","lasso","svr","rf"
    ], default="ridge")
    p.add_argument("--mlp_epochs",      type=int,   default=100)
    p.add_argument("--mlp_lr",          type=float, default=1e-3)
    p.add_argument("--mlp_patience",    type=int,   default=10)
    p.add_argument("--test_frac",       type=float, default=0.2)
    p.add_argument("--cv_folds",        type=int,   default=5)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--out_suffix",      default="embeddings_age_classifier")
    args = p.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_age_prediction(args)