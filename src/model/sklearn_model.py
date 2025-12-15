from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from typing import Dict, Tuple
import pandas as pd

class Classifier():

    metrics = ["accuracy", "f1", "precision", "recall"]
    model = None

    def build_pipeline(self, stages: Tuple[str, object], model: object, **params) -> Pipeline:
        classifier = model(**params)
        pipeline = Pipeline([stages, ("clf", classifier)])
        return pipeline

    def train(self, train_X, train_y, pipeline):
        print("Treinando modelo...")
        self.model = pipeline.fit(train_X, train_y)
        return self.model
    
    def evaluate(self, test_X, test_y) -> Dict[str, float]:
        print("Fazendo predições...")
        predictions = self.model.predict(test_X)
        
        print("Avaliando...")
        results = {}
        results["accuracy"] = accuracy_score(test_y, predictions)
        results["f1"] = f1_score(test_y, predictions, average="weighted")
        results["precision"] = precision_score(test_y, predictions, average="weighted")
        results["recall"] = recall_score(test_y, predictions, average="weighted")

        cm = confusion_matrix(test_y, predictions)
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv("/tmp/confusion_matrix_classification.csv", index=False)
                
        return results

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