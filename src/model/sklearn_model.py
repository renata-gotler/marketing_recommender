from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from typing import Dict, Tuple, Any
import pandas as pd

class Classifier():
    """
    A generic classifier wrapper for building pipelines, training, evaluating, and tuning classification models.
    """

    metrics = [f1_score, precision_score, recall_score]
    model: Any = None

    def build_pipeline(self, stages: Tuple[str, object], model: object, **params) -> Pipeline:
        """
        Constructs a scikit-learn pipeline with the provided preprocessing stages and classifier.

        Args:
            stages (Tuple[str, object]): Preprocessing stage as (name, transformer).
            model (object): Classifier class (not instance).
            **params: Parameters for the classifier.

        Returns:
            Pipeline: Configured scikit-learn pipeline.
        """
        classifier = model(**params)
        pipeline = Pipeline([stages, ("clf", classifier)])
        return pipeline

    def train(self, train_X: Any, train_y: Any, pipeline: Pipeline) -> Any:
        """
        Trains the pipeline on the provided training data.

        Args:
            train_X (Any): Training features.
            train_y (Any): Training labels.
            pipeline (Pipeline): scikit-learn pipeline.

        Returns:
            Any: Fitted pipeline.
        """
        print("Treinando modelo...")
        self.model = pipeline.fit(train_X, train_y)
        return self.model
    
    def evaluate(self, predictions: Any, test_y: Any) -> Dict[str, float]:
        """
        Evaluates predictions using accuracy and multiple classification metrics.

        Args:
            predictions (Any): Predicted labels.
            test_y (Any): True labels.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        print("Fazendo predições...")        
        print("Avaliando...")
        results: Dict[str, float] = {}
        results["accuracy"] = accuracy_score(test_y, predictions)
        for metric in self.metrics:
            for average in ["macro", 'weighted', "micro"]:
                metric_name = f"{metric.__name__}_{average}"
                results[metric_name] = metric(test_y, predictions, average=average)             
        return results

    def get_confusion_matrix(self, predictions: Any, test_y: Any, **params) -> Any:
        """
        Computes and saves the confusion matrix as a CSV file.

        Args:
            predictions (Any): Predicted labels.
            test_y (Any): True labels.
            **params: Additional parameters for confusion_matrix.

        Returns:
            Any: Confusion matrix array.
        """
        cm = confusion_matrix(test_y, predictions, **params)
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv("/tmp/confusion_matrix_classification.csv", index=False)
        return cm

    def cross_validation_tuning(
        self,
        train_X: Any,
        train_y: Any,
        base_pipeline: Pipeline,
        paramGrid: Dict[str, Any],
        evaluator: str = "f1",
        numFolds: int = 3,
        seed: int = 42
    ) -> GridSearchCV:
        """
        Performs cross-validation and hyperparameter tuning using GridSearchCV.

        Args:
            train_X (Any): Training features.
            train_y (Any): Training labels.
            base_pipeline (Pipeline): scikit-learn pipeline.
            paramGrid (Dict[str, Any]): Hyperparameter grid.
            evaluator (str, optional): Scoring metric. Defaults to "f1".
            numFolds (int, optional): Number of folds. Defaults to 3.
            seed (int, optional): Random seed. Defaults to 42.

        Returns:
            GridSearchCV: Fitted GridSearchCV object.
        """
        print("\n=== Realizando Cross-Validation e Tuning ===")
        
        cv = GridSearchCV(
            estimator=base_pipeline,
            param_grid=paramGrid,
            scoring=evaluator,
            cv=numFolds
        )

        print("Iniciando cross-validation... (pode demorar)")
        cv_model = cv.fit(train_X, train_y)
        
        return cv_model