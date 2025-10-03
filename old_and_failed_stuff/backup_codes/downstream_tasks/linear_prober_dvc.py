import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import RidgeClassifier, LogisticRegression, Ridge, Lasso
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, mean_squared_error, multilabel_confusion_matrix
from sklearn.metrics import classification_report
import json
import os
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

def preprocess_data(X):
    X = X.astype(np.float64)
    X = np.where(np.isinf(X), np.nan, X)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    X = np.clip(X, -65504, 65504)
    return X

class LinearProber:
    def __init__(
        self,
        seeds: List[int] = [41, 42, 43],
        test_size: float = 0.2,
        cv_folds: int = 2
    ):
        self.embeddings_path = "dvc-data/outputs/experiment_sub20CA/test_submission_combined.npy"
        self.labels_path = "dvc-data/arrays_sub20_with_cage_complete_correct_strains.npy"
        self.output_dir = "dvc-data/test-outputs"
        self.seeds = seeds
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.grouping_variable = "Strain"  # set grouping variable HERE

        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data()

        self.classifiers = {
            'RidgeClassifier': {
                'model': RidgeClassifier(class_weight='balanced'),
                'params': {'alpha': np.logspace(-1, 10, 10)}
            },
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=1000, class_weight='balanced'),
                'params': {'C': np.logspace(-5, 5, 10)}
            }
        }
        self.regressors = {
            'Ridge': {
                'model': Ridge(),
                'params': {'alpha': np.logspace(-1, 20, 10)}
            },
            'Lasso': {
                'model': Lasso(max_iter=10000),
                'params': {'alpha': np.logspace(-1, 20, 10)}
            }
        }

    def load_data(self) -> None:
        data = np.load(self.embeddings_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.size == 1:
            data = data.item()
        self.embeddings = data['embeddings']
        labels = np.load(self.labels_path, allow_pickle=True)
        if isinstance(labels, np.ndarray) and labels.size == 1:
            labels = labels.item()
        self.label_array = labels['label_array']
        self.vocabulary = labels['vocabulary']
        self.task_types = labels['task_type']

        # Extract grouping array for later use
        self.grouping_array = self.label_array[self.vocabulary.index(self.grouping_variable)]

    def train_eval_single_task(
        self, task_idx: int, seed: int, pdf: PdfPages
    ) -> Tuple[str, Dict[str, float]]:
        y = self.label_array[task_idx]
        indices = np.arange(len(y))
        train_idx, test_idx, y_train, y_test = train_test_split(
            indices, y,
            test_size=self.test_size,
            random_state=seed
        )

        X_train = preprocess_data(self.embeddings[train_idx])
        X_test = preprocess_data(self.embeddings[test_idx])
        task_type = self.task_types[task_idx].lower()

        algos = self.classifiers if task_type in ['discrete', 'classification'] else self.regressors

        best_algo, best_params, best_score = None, None, -np.inf
        eval_metrics = {}

        for name, cfg in algos.items():
            model = cfg['model']
            param_grid = cfg['params']
            param_name = list(param_grid.keys())[0]
            scoring = 'f1_macro' if task_type in ['discrete', 'classification'] else 'neg_root_mean_squared_error'
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=scoring,
                cv=self.cv_folds,
                n_jobs=-1,
                return_train_score=False
            )
            gs.fit(X_train, y_train)

            score = gs.best_score_
            print(f"{name} best {scoring}: {score:.4f} | Params: {gs.best_params_}")

            # Plotting hyperparameter performance
            plt.figure()
            x = param_grid[param_name]
            y_vals = gs.cv_results_['mean_test_score']
            plt.semilogx(x, y_vals, marker='o')
            plt.xlabel(param_name)
            plt.ylabel(scoring)
            plt.title(f"Task: {self.vocabulary[task_idx]} | Model: {name}")
            plt.grid(True)
            pdf.savefig()
            plt.close()

            if score > best_score:
                best_score = score
                best_algo = name
                best_params = gs.best_params_

        print(f"Selected {best_algo} with params {best_params}\n")

        final_model = algos[best_algo]['model'].set_params(**best_params)
        final_model.fit(X_train, y_train)
        y_pred_test = final_model.predict(X_test)

        grouping_test = self.grouping_array[test_idx]
        unique_grouping = np.unique(grouping_test)

        if task_type in ['discrete', 'classification']:
            f1 = f1_score(y_test, y_pred_test, average='macro')
            cms = multilabel_confusion_matrix(y_test, y_pred_test)
            tn = fp = fn = tp = 0
            for cm in cms:
                tn += cm[0, 0]; fp += cm[0, 1]
                fn += cm[1, 0]; tp += cm[1, 1]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            eval_metrics = {
                'f1_macro': f1,
                'precision': precision,
                'recall': recall
            }

            if self.grouping_variable == "Strain":
                # Per-group F1 Macro
                report = classification_report(y_test, y_pred_test, output_dict=True)
                grouping_f1 = {
                    label: metrics['f1-score']
                    for label, metrics in report.items()
                    if label not in ['accuracy', 'macro avg', 'weighted avg']
                }
                eval_metrics[f'{self.grouping_variable}_f1_macro'] = grouping_f1

                # Plot per-group F1 Macro
                grouping_f1 = dict(sorted(grouping_f1.items(), key=lambda item: item[1], reverse=True))
                plt.figure(figsize=(12, 6))
                plt.bar(grouping_f1.keys(), grouping_f1.values())
                plt.xlabel(f"{self.grouping_variable} ID")
                plt.xticks(rotation=45, ha='right', fontsize=5)
                plt.ylabel("F1 Macro")
                plt.title(f"Per-{self.grouping_variable} F1 Macro for {self.vocabulary[task_idx]}")
                plt.grid(True)
                pdf.savefig()
                plt.close()


                # Count number of incorrect predictions per true strain
                error_matrix = defaultdict(lambda: defaultdict(int))

                for true_label, pred_label in zip(y_test, y_pred_test):
                    if true_label != pred_label:
                        error_matrix[true_label][pred_label] += 1

                # Extract all unique strains involved
                all_strains = sorted(set(error_matrix.keys()) | {pred for preds in error_matrix.values() for pred in preds})

                # Build stacked bar plot
                # Build stacked bar plot (original version)
                strain_order = sorted(error_matrix.keys(), key=lambda s: sum(error_matrix[s].values()), reverse=True)
                all_strains = sorted(set(error_matrix.keys()) | {pred for preds in error_matrix.values() for pred in preds})

                bar_data = {
                    pred: [error_matrix[true].get(pred, 0) if pred != true else 0 for true in strain_order]
                    for pred in all_strains
                }

                # Original (unnormalized)
                plt.figure(figsize=(14, 6))
                bottom = np.zeros(len(strain_order))
                for pred_strain, counts in bar_data.items():
                    plt.bar(strain_order, counts, bottom=bottom, label=f"{pred_strain}")
                    bottom += np.array(counts)

                plt.xlabel("True Strain")
                plt.ylabel("Count of Incorrect Predictions")
                plt.title(f"Stacked Incorrect Predictions per Strain for {self.vocabulary[task_idx]}")
                plt.xticks(rotation=45, ha='right', fontsize=6)
                plt.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left', title=None)
                plt.tight_layout()
                plt.grid(True, axis='y')
                pdf.savefig()
                plt.close()

                # Normalized version (per-bar percentage)
                bar_data_normalized = {
                    pred: [val / sum_vals if sum_vals > 0 else 0
                        for val, sum_vals in zip(counts, bottom)]
                    for pred, counts in bar_data.items()
                }

                plt.figure(figsize=(14, 6))
                bottom_norm = np.zeros(len(strain_order))
                for pred_strain, counts in bar_data_normalized.items():
                    plt.bar(strain_order, counts, bottom=bottom_norm, label=f"{pred_strain}")
                    bottom_norm += np.array(counts)

                plt.xlabel("True Strain")
                plt.ylabel("Proportion of Incorrect Predictions")
                plt.title(f"Normalized Stacked Incorrect Predictions per Strain for {self.vocabulary[task_idx]}")
                plt.xticks(rotation=45, ha='right', fontsize=6)
                plt.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left', title=None)
                plt.tight_layout()
                plt.grid(True, axis='y')
                pdf.savefig()
                plt.close()


        else:
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            eval_metrics = {
                'mse': mse,
                'rmse': rmse
            }


            # Per-group RMSE using grouping variable
            grouping_test = self.grouping_array[test_idx]
            unique_grouping = np.unique(grouping_test)
            grouping_rmse = {}

            for group in unique_grouping:
                idxs = np.where(grouping_test == group)[0]
                if len(idxs) == 0:
                    continue
                mse_group = mean_squared_error(y_test[idxs], y_pred_test[idxs])
                rmse_group = np.sqrt(mse_group)
                grouping_rmse[group] = rmse_group

            eval_metrics[f'{self.grouping_variable}_rmse'] = grouping_rmse

            grouping_rmse = dict(sorted(grouping_rmse.items(), key=lambda item: item[1], reverse=True))

            plt.figure(figsize=(12, 6))
            plt.bar(grouping_rmse.keys(), grouping_rmse.values())
            plt.xlabel(f"{self.grouping_variable} ID")
            plt.xticks(rotation=45, ha='right', fontsize=5)
            plt.ylabel("RMSE")
            plt.title(f"Per-{self.grouping_variable} RMSE for {self.vocabulary[task_idx]}")
            plt.grid(True)
            pdf.savefig()
            plt.close()


        return best_algo, {'params': best_params, **eval_metrics}

    def evaluate_all_tasks(self) -> Dict:
        results = {}
        out_path = os.path.join(self.output_dir, "linear_probing_results_sub20CA_Stacked.json")
        pdf_path = out_path.replace(".json", ".pdf")

        with PdfPages(pdf_path) as pdf:
            for idx, name in enumerate(self.vocabulary[:-1]):
                print(f"\nEvaluating task {idx}: {name}")
                algo, metrics = self.train_eval_single_task(idx, seed=self.seeds[0], pdf=pdf)
                results[name] = {
                    'task_type': self.task_types[idx],
                    'best_algorithm': algo,
                    **metrics
                }

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

if __name__ == '__main__':
    prober = LinearProber(
        seeds=[41, 42, 43],
        test_size=0.2,
        cv_folds=3
    )
    start_time = time.time()
    results = prober.evaluate_all_tasks()
    for task, res in tqdm(results.items(), desc="Evaluating tasks", total=len(results)):
        print(f"Task: {task}")
        print(f"Type: {res['task_type']}, Algo: {res['best_algorithm']}, Params: {res['params']}")
        for m, v in res.items():
            if m not in ['task_type', 'best_algorithm', 'params']:
                if isinstance(v, dict):
                    print(f"  {m}: {{...}} ({len(v)} items)")
                else:
                    print(f"  {m}: {v:.4f}")
        print()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Linear prober time {}".format(total_time_str))
