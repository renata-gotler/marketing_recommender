
from typing import List, Any, Tuple, Optional, Dict

from pyspark.ml.feature import StringIndexer, OneHotEncoder, StandardScaler, Imputer, VectorAssembler
from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when

class StrImputer(Transformer):
    """
    Custom Transformer to impute missing values in the 'gender' column with a default value.
    """
    def __init__(self, inputCol: str, outputCols: str, value: str) -> None:
        """
        Initialize StrImputer.

        Args:
            inputCol (str): Name of the input column.
        """
        super().__init__()
        self.inputCol = inputCol
        self.outputCols = outputCols
        self.default_value = value

    def _transform(self, df: DataFrame) -> DataFrame:
        """
        Transform the DataFrame by imputing missing values in the string column.

        Args:
            df (DataFrame): Input Spark DataFrame.

        Returns:
            DataFrame: Transformed DataFrame with imputed gender values.
        """
        return df.withColumn(
            self.outputCols,
            when(col(self.inputCol).isNull(), self.default_value)
             .otherwise(col(self.inputCol))
        )

class PreProcessing():
    """
    Class for building preprocessing pipelines for categorical and numerical features.
    """

    def __init__(self) -> None:
        """
        Initialize PreProcessing with empty pipeline stages and feature lists.
        """
        self.stages: List[Any] = []
        self.encoder_cols: List[str] = []
        self.numerical_cols: List[str] = []

    def build_categorical_transformation_pipeline(self, categorical_features: List[str], imputer_mapping: Optional[Dict[str, str]] = None) -> None:
        """
        Build transformation pipeline for categorical features including imputation, indexing, and encoding.

        Args:
            categorical_features (List[str]): List of categorical feature names.
            imputer_mapping (Optional[Dict[str, str]], optional): Mapping of columns to default values for imputation. Defaults to None.
        """
        if imputer_mapping:
            for col_name, value in imputer_mapping.items():
                imputer: StrImputer = StrImputer(inputCol=col_name, outputCols=col_name, value=value)
                self.stages.append(imputer)

        for cat_col in categorical_features:
            indexer: StringIndexer = StringIndexer(
                inputCol=cat_col,
                outputCol=f"{cat_col}_idx",
                handleInvalid="keep"
            )
            self.stages.append(indexer)
        
        for cat_col in categorical_features:
            encoder: OneHotEncoder = OneHotEncoder(
                inputCol=f"{cat_col}_idx",
                outputCol=f"{cat_col}_vec"
            )
            self.stages.append(encoder)
            self.encoder_cols.append(f"{cat_col}_vec")

    def build_numerical_transformation_pipeline(self, numerical_features: List[str]) -> None:
        """
        Build transformation pipeline for numerical features including imputation and scaling.

        Args:
            numerical_features (List[str]): List of numerical feature names.
        """
        imputer = Imputer(
            inputCols=numerical_features,
            outputCols=[f"{c}_imputed" for c in numerical_features],
            strategy="median"
        )
        self.stages.append(imputer)

        for num_col in numerical_features:
            scaler: StandardScaler = StandardScaler(inputCol=f"{num_col}_imputed", outputCol=f"{num_col}_scaled")
            self.stages.append(scaler)
            self.numerical_cols.append(f"{num_col}_scaled")
        
    def index_target(self, target_col: str) -> None:
        """
        Index the target column for classification tasks.

        Args:
            target_col (str): Name of the target column.
        """
        target_indexer: StringIndexer = StringIndexer(
            inputCol=target_col,
            outputCol="label",
            handleInvalid="keep"
        )
        self.stages.append(target_indexer)

    def assemble_pipeline(self) -> None:
        """
        Assemble the final pipeline by combining all feature columns into a single vector.
        """
        feature_cols: List[str] = self.numerical_cols + self.encoder_cols
        assembler: Any = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        self.stages.append(assembler)