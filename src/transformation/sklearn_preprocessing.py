from typing import List, Any, Tuple, Optional, Dict
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np


class PreProcessing:
    """
    Class for building preprocessing pipelines for categorical and numerical features.
    """

    def __init__(self) -> None:
        self.categorical_transformers: List[Tuple[str, Any, List[str]]] = []
        self.numerical_transformers: List[Tuple[str, Any, List[str]]] = []
        self.feature_names: List[str] = []
        self.target_encoder: Optional[LabelEncoder] = None
        self.stages: Tuple[str, object] = None

    def build_categorical_transformation_pipeline(
        self, categorical_features: List[str], imputer_mapping: Optional[Dict[str, str]] = None
    ) -> None:
        for col in categorical_features:
            fill_value = imputer_mapping.get(col, "missing")
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=fill_value)),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])
            self.categorical_transformers.append((f"{col}_cat", cat_pipeline, [col]))


    def build_numerical_transformation_pipeline(self, numerical_features: List[str]) -> None:
        if numerical_features:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            self.numerical_transformers.append(("num", num_pipeline, numerical_features))

    def index_target(self, target_col: str) -> None:
        self.target_encoder = LabelEncoder()
        self.target_col = target_col

    def assemble_pipeline(self) -> None:
        transformers = []
        if self.categorical_transformers:
            transformers.extend(self.categorical_transformers)
        if self.numerical_transformers:
            transformers.extend(self.numerical_transformers)
        preprocessor = ColumnTransformer(transformers)
        self.stages = ("preprocessing", preprocessor)