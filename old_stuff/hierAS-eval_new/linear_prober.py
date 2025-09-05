import numpy as np
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
import json
import os
from typing import Dict, List, Tuple, Union
from sklearn.metrics import multilabel_confusion_matrix

#takes care of large values appearing in the embeddings, also for raw data changes it to numeric type
def preprocess_data(X):
    # Ensure the data is of a numeric type
    X = X.astype(np.float64)
    # Replace inf values with NaN
    X = np.where(np.isinf(X), np.nan, X)
    # Impute NaN values with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    # Clip values to a reasonable range for float16
    X = np.clip(X, -65504, 65504)
    return X

class LinearProber:
    def __init__(
        self, 
        embeddings_path: str,
        labels_path: str,
        output_dir: str,
        seeds: List[int] = [41, 42, 43],
        test_size: float = 0.2
    ):
        """Initialize linear probing setup
        
        Args:
            embeddings_path: Path to .npy file with frame embeddings
            labels_path: Path to .npy file with frame labels
            output_dir: Where to save results
            seeds: Random seeds for multiple runs
            test_size: Fraction of data to use for testing
        """
        self.embeddings_path = "dvc-data/outputs/experiment5/test_submission_combined.npy"
        self.labels_path = "dvc-data/day_night_strain_sub.npy"
        self.output_dir = "dvc-data/test-outputs"
        self.seeds = seeds
        self.test_size = test_size
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        self.load_data()

    def load_data(self) -> None:
        """Load embeddings and labels"""
        # Load embeddings allowing us to load raw data
        data = np.load(self.embeddings_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.size == 1:
            data = data.item()
        self.embeddings = data["embeddings"] # = data for raw data, = data["embeddings"] for embeddings
        
        # Load labels
        labels = np.load(self.labels_path, allow_pickle=True)
        if isinstance(labels, np.ndarray) and labels.size == 1:
            labels = labels.item()
        self.label_array = labels["label_array"]
        
        
        self.vocabulary = labels["vocabulary"]
        self.task_types = labels["task_type"]

    def train_eval_single_task(
        self, 
        task_idx: int, 
        seed: int
    ) -> Tuple[float, float]:
        """Train and evaluate a single task with one seed
        
        Args:
            task_idx: Index of the task/label to train on
            seed: Random seed
            
        Returns:
            train_score: Score on training set
            test_score: Score on test set
        """
        # Get labels for this task
        y = self.label_array[task_idx]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.embeddings, y, 
            test_size=self.test_size,
            random_state=seed
        )
        
        # Initialize model based on task type
        if self.task_types[task_idx] == "Discrete":
            model = RidgeClassifier(alpha=1.0, class_weight="balanced")
            score_fn = lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
        else:
            model = Ridge(alpha=1.0)
            score_fn = mean_squared_error
        X_train = preprocess_data(X_train)
        X_test = preprocess_data(X_test)
        # Train
        model.fit(X_train, y_train)

        # Evaluate
        train_score = score_fn(y_train, model.predict(X_train))
        test_score = score_fn(y_test, model.predict(X_test))

        # Calculate confusion matrix
        y_pred = model.predict(X_test)
        confusion_matrices = multilabel_confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = [], [], [], []
        for cm in confusion_matrices:
            tn.append(cm[0, 0])
            fp.append(cm[0, 1])
            fn.append(cm[1, 0])
            tp.append(cm[1, 1])

        # Sum up the values for all labels
        tn = np.sum(tn)
        fp = np.sum(fp)
        fn = np.sum(fn)
        tp = np.sum(tp)

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return train_score, test_score, precision, recall

    def evaluate_all_tasks(self) -> Dict:
        """Evaluate all tasks across all seeds
        
        Returns:
            results: Dictionary with results for each task and seed
        """
        results = {}
        
        for task_idx, task_name in enumerate(self.vocabulary):
            task_results = {
                "train_scores": [],
                "test_scores": [],
                "precision": [],
                "recall": [],
                "task_type": self.task_types[task_idx]
            }
            
            #added precision and recall to the output for better understanding (linked to false positives and negatives)
            for seed in self.seeds:
                train_score, test_score, precision, recall = self.train_eval_single_task(task_idx, seed)
                task_results["train_scores"].append(train_score)
                task_results["test_scores"].append(test_score)
                task_results["precision"].append(precision)
                task_results["recall"].append(recall)
            
            # Calculate mean and std
            task_results["mean_train"] = float(np.mean(task_results["train_scores"]))
            task_results["std_train"] = float(np.std(task_results["train_scores"]))
            task_results["mean_test"] = float(np.mean(task_results["test_scores"]))
            task_results["std_test"] = float(np.std(task_results["test_scores"]))
        
            results[task_name] = task_results
            
        # Save results
        output_path = os.path.join(self.output_dir, "linear_probing_results_exp5.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
        return results



if __name__ == "__main__":

    # Initialize probing
    prober = LinearProber(
        embeddings_path="path/to/embeddings.npy",
        labels_path="path/to/labels.npy", 
        output_dir="./linear_probing_results",
        seeds=[41, 42, 43],
        test_size=0.2
    )

    # Run evaluation
    results = prober.evaluate_all_tasks()

    # Print results
    for task_name, task_results in results.items():
        print(f"\nTask: {task_name}")
        print(f"Type: {task_results['task_type']}")
        print(f"Mean test score: {task_results['mean_test']:.3f} ± {task_results['std_test']:.3f}")

