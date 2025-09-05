import argparse
import json
import os
import time
import datetime

import numpy as np
import pandas as pd
import torch # For custom MLP
import torch.nn as nn # For custom MLP
import torch.optim as optim # For custom MLP
from sklearn.model_selection import train_test_split, GridSearchCV, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# (preprocess_embeddings function can be shared or duplicated)
def preprocess_embeddings(X):
    X_clean = X.astype(np.float64)
    X_clean = np.where(np.isinf(X_clean), np.nan, X_clean)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_clean)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, scaler

class SimpleMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.3):
        super(SimpleMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)
        # For CrossEntropyLoss, no softmax here. If using NLLLoss, add LogSoftmax.

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x

def train_mlp_classifier(X_train, y_train_encoded, X_val, y_val_encoded, input_dim, num_classes, epochs=100, lr=0.001, batch_size=32, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train_encoded))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val_encoded))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleMLPClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss() # Good for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1_macro': []}

    for epoch in tqdm(range(epochs), desc="Training MLP Classifier"):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0
        all_val_preds = []
        all_val_true = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y_true = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y_true)
                epoch_val_loss += loss.item()
                _, predicted_labels = torch.max(outputs, 1)
                all_val_preds.extend(predicted_labels.cpu().numpy())
                all_val_true.extend(batch_y_true.cpu().numpy())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_f1 = f1_score(all_val_true, all_val_preds, average='macro', zero_division=0)
        history['val_loss'].append(avg_val_loss)
        history['val_f1_macro'].append(val_f1)
        
        # print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val F1 Macro: {val_f1:.4f}")

        if avg_val_loss < best_val_loss: # Or monitor val_f1 (higher is better)
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_mlp_strain_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load("best_mlp_strain_model.pth"))
    return model, history

def evaluate_mlp_classifier(model, X_test, y_test_encoded, le, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test_encoded))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_test_preds_encoded = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted_labels = torch.max(outputs, 1)
            all_test_preds_encoded.extend(predicted_labels.cpu().numpy())
            
    y_pred_test_original = le.inverse_transform(all_test_preds_encoded)
    y_test_original = le.inverse_transform(y_test_encoded)

    f1_macro = f1_score(y_test_original, y_pred_test_original, average='macro', zero_division=0)
    report = classification_report(y_test_original, y_pred_test_original, output_dict=True, zero_division=0, labels=le.classes_)
    
    metrics = {'f1_macro': f1_macro, 'classification_report': report}
    return metrics, y_pred_test_original


def run_strain_prediction(args):
    print(f"--- Running Strain Prediction (Classifier: {args.classifier_type}) ---")
    output_file_prefix = f"strain_prediction_{args.classifier_type}_{args.embeddings_suffix}"
    results_json_path = os.path.join(args.output_dir, f"{output_file_prefix}_results.json")
    pdf_path = os.path.join(args.output_dir, f"{output_file_prefix}_plots.pdf")

    # 1. Load Embeddings
    print(f"Loading embeddings from: {args.embeddings_path}")
    data = np.load(args.embeddings_path, allow_pickle=True).item()
    embeddings = data['embeddings']
    frame_map = data['frame_number_map']

    # 2. Load Labels (Strain and Cage for grouping)
    print(f"Loading labels from: {args.labels_path}")
    labels_data = np.load(args.labels_path, allow_pickle=True).item()
    label_array = labels_data['label_array']
    vocabulary = labels_data['vocabulary']

    try:
        strain_idx = vocabulary.index("Strain")
        cage_idx = vocabulary.index("Cage") # For grouped split
    except ValueError:
        print("Error: 'Strain' or 'Cage' not found in labels vocabulary.")
        return

    strain_labels_all_frames = label_array[strain_idx].astype(str) # Ensure string type for LE
    cage_ids_all_frames = label_array[cage_idx]

    # Aggregate embeddings and labels per day (similar to age prediction)
    daily_embeddings = []
    daily_strain_labels = []
    daily_cage_ids = []

    print("Aggregating embeddings per day for strain...")
    unique_seq_ids = sorted(list(frame_map.keys()))
    for seq_id in tqdm(unique_seq_ids):
        start_idx, end_idx = frame_map[seq_id]
        if start_idx == end_idx: continue

        daily_embeddings.append(np.mean(embeddings[start_idx:end_idx], axis=0))
        daily_strain_labels.append(strain_labels_all_frames[start_idx])
        daily_cage_ids.append(cage_ids_all_frames[start_idx])

    X_daily = np.array(daily_embeddings)
    y_daily_strain_original = np.array(daily_strain_labels)
    groups_daily_cage = np.array(daily_cage_ids)
    
    # Encode string labels to integers
    le = LabelEncoder()
    y_daily_strain_encoded = le.fit_transform(y_daily_strain_original)
    num_classes = len(le.classes_)
    print(f"Strain classes: {le.classes_} (Encoded: {np.unique(y_daily_strain_encoded)})")


    X_processed, _ = preprocess_embeddings(X_daily)

    # GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_set_fraction, random_state=args.seed)
    train_val_idx, test_idx = next(gss.split(X_processed, y_daily_strain_encoded, groups_daily_cage))

    X_train_val, X_test = X_processed[train_val_idx], X_processed[test_idx]
    y_train_val_enc, y_test_enc = y_daily_strain_encoded[train_val_idx], y_daily_strain_encoded[test_idx]
    groups_train_val = groups_daily_cage[train_val_idx]

    print(f"Train_val samples: {X_train_val.shape[0]}, Test samples: {X_test.shape[0]}")

    overall_results = {"args": vars(args), "model_type": args.classifier_type}
    pdf = PdfPages(pdf_path)
    y_pred_test_original_labels = None


    if args.classifier_type == 'mlp':
        gss_mlp = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        train_idx_mlp, val_idx_mlp = next(gss_mlp.split(X_train_val, y_train_val_enc, groups_train_val))
        
        X_train_mlp, X_val_mlp = X_train_val[train_idx_mlp], X_train_val[val_idx_mlp]
        y_train_mlp_enc, y_val_mlp_enc = y_train_val_enc[train_idx_mlp], y_train_val_enc[val_idx_mlp]

        print(f"MLP Train samples: {X_train_mlp.shape[0]}, MLP Val samples: {X_val_mlp.shape[0]}")

        mlp_model, history = train_mlp_classifier(X_train_mlp, y_train_mlp_enc, X_val_mlp, y_val_mlp_enc,
                                                  input_dim=X_processed.shape[1], num_classes=num_classes,
                                                  epochs=args.mlp_epochs, lr=args.mlp_lr,
                                                  batch_size=args.mlp_batch_size, patience=args.mlp_patience)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('MLP Loss for Strain Prediction')
        plt.xlabel('Epochs'); plt.ylabel('CrossEntropy Loss'); plt.legend(); plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['val_f1_macro'], label='Validation F1 Macro')
        plt.title('MLP Validation F1 Macro for Strain')
        plt.xlabel('Epochs'); plt.ylabel('F1 Macro'); plt.legend(); plt.grid(True)
        pdf.savefig(); plt.close()

        test_metrics_dict, y_pred_test_original_labels = evaluate_mlp_classifier(mlp_model, X_test, y_test_enc, le, batch_size=args.mlp_batch_size)
        overall_results['test_metrics'] = test_metrics_dict

    elif args.classifier_type == 'custom':
        print("Custom classifier training not implemented yet.")
        pdf.close(); return
    else: # Sklearn linear models
        classifiers = {
            'LogisticRegression': {'model': LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear'), 
                                   'params': {'C': np.logspace(-3, 3, 7)}},
            'RidgeClassifier': {'model': RidgeClassifier(class_weight='balanced'), 
                                'params': {'alpha': np.logspace(-3, 3, 7)}},
            'SVC': {'model': SVC(class_weight='balanced', probability=True), # probability for potential calibration later
                    'params': {'C': np.logspace(-2,2,5), 'gamma': ['scale', 'auto'] + list(np.logspace(-3,0,4))}},
            'RandomForestClassifier': {'model': RandomForestClassifier(random_state=args.seed, class_weight='balanced', n_jobs=-1),
                                     'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}}
        }
        if args.classifier_type not in classifiers:
            raise ValueError(f"Unsupported classifier_type: {args.classifier_type}")

        cfg = classifiers[args.classifier_type]
        model_instance = cfg['model']
        param_grid = cfg['params']
        
        gss_cv = GroupShuffleSplit(n_splits=args.cv_folds, test_size=0.25, random_state=args.seed)
        gs = GridSearchCV(
            estimator=model_instance, param_grid=param_grid, scoring='f1_macro',
            cv=list(gss_cv.split(X_train_val, y_train_val_enc, groups_train_val)),
            n_jobs=-1, verbose=1
        )
        gs.fit(X_train_val, y_train_val_enc)

        print(f"Best {args.classifier_type} F1 Macro (CV): {gs.best_score_:.4f}")
        print(f"Best Params: {gs.best_params_}")

        if len(param_grid) == 1: # Simple plot for one hyperparameter
            param_name = list(param_grid.keys())[0]
            plt.figure()
            plt.semilogx(param_grid[param_name], gs.cv_results_['mean_test_score'], marker='o')
            plt.xlabel(param_name); plt.ylabel("F1 Macro"); plt.grid(True)
            plt.title(f"Hyperparameter Tuning for {args.classifier_type} (Strain)")
            pdf.savefig(); plt.close()

        final_model = gs.best_estimator_
        final_model.fit(X_train_val, y_train_val_enc)
        y_pred_test_enc = final_model.predict(X_test)
        y_pred_test_original_labels = le.inverse_transform(y_pred_test_enc)
        
        f1_macro_test = f1_score(le.inverse_transform(y_test_enc), y_pred_test_original_labels, average='macro', zero_division=0, labels=le.classes_)
        report_test = classification_report(le.inverse_transform(y_test_enc), y_pred_test_original_labels, output_dict=True, zero_division=0, labels=le.classes_)
        overall_results['test_metrics'] = {'f1_macro': f1_macro_test, 'classification_report': report_test}
        overall_results['best_params'] = gs.best_params_

    # --- Common for all models: Confusion Matrix ---
    if y_pred_test_original_labels is not None:
        cm = confusion_matrix(le.inverse_transform(y_test_enc), y_pred_test_original_labels, labels=le.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        fig, ax = plt.subplots(figsize=(10, 8)) # Adjust size as needed
        disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', colorbar=True)
        plt.title(f"Confusion Matrix for Strain Prediction ({args.classifier_type})")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # --- Per-Group F1 (using the specified group_analysis_variable, e.g., Cage) ---
        if args.group_analysis_variable:
            try:
                group_var_idx = vocabulary.index(args.group_analysis_variable)
                group_labels_all_frames = label_array[group_var_idx]
                daily_group_var_labels = []
                for seq_id in unique_seq_ids: # Assuming unique_seq_ids matches daily_embeddings order
                    start_idx, end_idx = frame_map[seq_id]
                    daily_group_var_labels.append(group_labels_all_frames[start_idx]) # One group label per day
                
                daily_group_var_labels = np.array(daily_group_var_labels)
                test_group_labels = daily_group_var_labels[test_idx] # Get group labels for the test set items
                
                unique_groups_in_test = np.unique(test_group_labels)
                per_group_f1 = {}
                for group_val in unique_groups_in_test:
                    group_mask_test = (test_group_labels == group_val)
                    if np.sum(group_mask_test) > 1 : # Need at least 2 samples for F1 usually
                        y_test_group_orig = le.inverse_transform(y_test_enc[group_mask_test])
                        y_pred_group_orig = y_pred_test_original_labels[group_mask_test]
                        per_group_f1[str(group_val)] = f1_score(y_test_group_orig, y_pred_group_orig, average='macro', zero_division=0, labels=le.classes_)
                
                overall_results[f'per_{args.group_analysis_variable}_f1_macro'] = dict(sorted(per_group_f1.items(), key=lambda item: item[1], reverse=True))
                
                if per_group_f1:
                    plt.figure(figsize=(12,6))
                    plt.bar(overall_results[f'per_{args.group_analysis_variable}_f1_macro'].keys(), overall_results[f'per_{args.group_analysis_variable}_f1_macro'].values())
                    plt.xlabel(f"{args.group_analysis_variable} ID")
                    plt.ylabel("F1 Macro")
                    plt.title(f"Per-{args.group_analysis_variable} F1 Macro for Strain Prediction ({args.classifier_type})")
                    plt.xticks(rotation=45, ha="right", fontsize=8)
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
            except ValueError:
                print(f"Warning: Group analysis variable '{args.group_analysis_variable}' not found in labels vocabulary.")


    pdf.close()
    print(f"Plots saved to {pdf_path}")

    with open(results_json_path, 'w') as f:
        json.dump(overall_results, f, indent=4, cls=NpEncoder)
    print(f"Results saved to {results_json_path}")

class NpEncoder(json.JSONEncoder): # Copied from age_classifier
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("DVC Strain Prediction from Embeddings")
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--embeddings_suffix", default="combined")
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--classifier_type", default="LogisticRegression", choices=['LogisticRegression', 'RidgeClassifier', 'SVC', 'RandomForestClassifier', 'mlp', 'custom'])
    parser.add_argument("--test_set_fraction", default=0.2, type=float)
    parser.add_argument("--cv_folds", default=3, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--mlp_epochs", default=100, type=int)
    parser.add_argument("--mlp_lr", default=0.001, type=float)
    parser.add_argument("--mlp_batch_size", default=32, type=int)
    parser.add_argument("--mlp_patience", default=10, type=int)
    parser.add_argument("--group_analysis_variable", default="Cage", type=str, help="Variable in labels to use for per-group analysis (e.g., Cage).")


    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_strain_prediction(args)