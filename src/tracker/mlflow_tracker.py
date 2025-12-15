
import mlflow


class MLFlowTracker:

    def __init__(self, experiment_name:str = "offer-predictions", tracking_uri: str = "databricks"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.setup_mlflow(experiment_name=experiment_name, tracking_uri=tracking_uri)


    def setup_mlflow(self, experiment_name:str, tracking_uri:str):
        """
        Configura MLflow para tracking
        """
        print("=== Configurando MLflow ===")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(f"/Shared/{self.experiment_name}")

    def load_model_from_registry(self, model_name: str, stage="Production"):
        """
        Carrega modelo do MLflow Model Registry
        """
        print(f"Carregando modelo {model_name} (stage: {stage})")
        
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.spark.load_model(model_uri)
        
        print(f"Modelo carregado com sucesso!")
        return model


    def load_model_from_run(self, run_id: str, artifact_path="model"):
        """
        Carrega modelo de um run específico
        """
        print(f"Carregando modelo do run {run_id}")
        
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model = mlflow.spark.load_model(model_uri)
        
        print(f"Modelo carregado com sucesso!")
        return model


    def get_best_model(self, metric: str = "f1", ascending=True):
        """
        Encontra o melhor modelo baseado em uma métrica
        """
        print(f"Buscando melhor modelo no experiment '{self.experiment_name}'...")
        
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            print(f"Experiment '{self.experiment_name}' não encontrado")
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if len(runs) == 0:
            print("Nenhum run encontrado")
            return None
        
        best_run = runs.iloc[0]
        print(f"\nMelhor run encontrado:")
        print(f"  Run ID: {best_run.run_id}")
        print(f"  {metric}: {best_run[f'metrics.{metric}']}")
        
        return best_run