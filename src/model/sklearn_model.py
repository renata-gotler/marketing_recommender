from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from typing import Dict, Tuple
import pandas as pd

class Classifier():

    metrics = [f1_score, precision_score, recall_score]
    model = None

    def build_pipeline(self, stages: Tuple[str, object], model: object, **params) -> Pipeline:
        classifier = model(**params)
        pipeline = Pipeline([stages, ("clf", classifier)])
        return pipeline

    def train(self, train_X, train_y, pipeline):
        print("Treinando modelo...")
        self.model = pipeline.fit(train_X, train_y)
        return self.model
    
    def evaluate(self, predictions, test_y) -> Dict[str, float]:
        print("Fazendo predições...")        
        print("Avaliando...")
        results = {}
        results["accuracy"] = accuracy_score(test_y, predictions)
        for metric in self.metrics:
            for average in ["macro", 'weighted', "micro"]:
                results[f"{metric.__name__}_{average}"] = metric(test_y, predictions, average=average)             
        return results

    def get_confusion_matrix(self, predictions, test_y, **params):
        cm = confusion_matrix(test_y, predictions, **params)
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv("/tmp/confusion_matrix_classification.csv", index=False)
        return cm

    def cross_validation_tuning(self, train_X, train_y, base_pipeline, paramGrid, evaluator="f1", numFolds: int = 3, seed: int = 42):
        """
        Realiza validação cruzada e tuning de hiperparâmetros com MLflow tracking
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