from typing import Dict

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator


class Classifier:
    metrics = ["accuracy", "f1", "weightedPrecision", "weightedRecall"]
    model = None

    def build_pipeline(self, stages: list, model: object, **params) -> Pipeline:
        classifier = model(**params)
        stages.append(classifier)
        pipeline = Pipeline(stages=stages)
        return pipeline

    def train(self, train_df, pipeline):
        print("Treinando modelo...")
        self.model = pipeline.fit(train_df)
        return self.model

    def evaluate(self, test_df) -> Dict[str, float]:
        print("Fazendo predições...")
        predictions = self.model.transform(test_df)

        print("Avaliando...")
        results = {}
        for metric in self.metrics:
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName=metric
            )
            results[metric] = evaluator.evaluate(predictions)

        confusion_matrix = predictions.groupBy("label", "prediction").count().toPandas()
        confusion_matrix.to_csv("/tmp/confusion_matrix_classification.csv", index=False)

        return results

    def cross_validation_tuning(
        self,
        train_df,
        base_pipeline,
        paramGrid,
        evaluator,
        numFolds: int = 3,
        seed: int = 42,
    ):
        """
        Realiza validação cruzada e tuning de hiperparâmetros com MLflow tracking
        """
        print("\n=== Realizando Cross-Validation e Tuning ===")

        cv = CrossValidator(
            estimator=base_pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=numFolds,
            seed=seed,
        )

        print("Iniciando cross-validation... (pode demorar)")
        cv_model = cv.fit(train_df)

        return cv_model
