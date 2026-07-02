import os
import sys
import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:

    transformed_data_dir: str = os.path.join(
        "artifacts",
        "transformed_data",
    )

    preprocessor_obj_path: str = os.path.join(
        transformed_data_dir,
        "preprocessor.pkl",
    )

    train_data_path: str = os.path.join(
        transformed_data_dir,
        "train_data.csv",
    )

    test_data_path: str = os.path.join(
        transformed_data_dir,
        "test_data.csv",
    )


class DataTransformation:

    def __init__(self):

        self.config = DataTransformationConfig()

    def initiate_data_transformation(
        self,
        df: pd.DataFrame,
    ):

        try:

            logging.info("=" * 80)
            logging.info("DATA TRANSFORMATION STARTED")
            logging.info("=" * 80)

            os.makedirs(
                self.config.transformed_data_dir,
                exist_ok=True,
            )

            #########################################################

            target_column = df.columns[-1]
            logging.info(f"Target Column : {target_column}")
            logging.info(
                f"Target Column : {target_column}"
            )

            #########################################################

            X = df.drop(columns=[target_column])

            y = df[target_column]

            #########################################################

            numeric_columns = X.select_dtypes(
                include=[np.number]
            ).columns.tolist()

            categorical_columns = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            logging.info(
                f"Numeric Columns : {numeric_columns}"
            )

            logging.info(
                f"Categorical Columns : {categorical_columns}"
            )

            #########################################################

            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler(),
                    )
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    )
                ]
            )

            #########################################################

            transformers = []

            if numeric_columns:

                transformers.append(
                    (
                        "num",
                        numeric_pipeline,
                        numeric_columns,
                    )
                )

            if categorical_columns:

                transformers.append(
                    (
                        "cat",
                        categorical_pipeline,
                        categorical_columns,
                    )
                )

            preprocessor = ColumnTransformer(
                transformers=transformers
            )

            #########################################################

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )

            logging.info(
                f"Train Shape : {X_train.shape}"
            )

            logging.info(
                f"Test Shape : {X_test.shape}"
            )

            #########################################################

            X_train = preprocessor.fit_transform(
                X_train
            )

            X_test = preprocessor.transform(
                X_test
            )

            #########################################################

            feature_names = []

            feature_names.extend(numeric_columns)

            if categorical_columns:

                encoded_features = (
                    preprocessor.named_transformers_[
                        "cat"
                    ]
                    .named_steps["encoder"]
                    .get_feature_names_out(
                        categorical_columns
                    )
                    .tolist()
                )

                feature_names.extend(encoded_features)

            #########################################################

            train_df = pd.DataFrame(
                X_train,
                columns=feature_names,
            )

            test_df = pd.DataFrame(
                X_test,
                columns=feature_names,
            )

            train_df[target_column] = (
                y_train.reset_index(drop=True)
            )

            test_df[target_column] = (
                y_test.reset_index(drop=True)
            )

            #########################################################

            train_df.to_csv(
                self.config.train_data_path,
                index=False,
            )

            test_df.to_csv(
                self.config.test_data_path,
                index=False,
            )

            #########################################################

            joblib.dump(
                preprocessor,
                self.config.preprocessor_obj_path,
            )

            #########################################################

            logging.info(
                "Preprocessor Saved Successfully."
            )

            logging.info(
                f"Train Data Saved : {self.config.train_data_path}"
            )

            logging.info(
                f"Test Data Saved : {self.config.test_data_path}"
            )

            logging.info(
                f"Preprocessor Saved : {self.config.preprocessor_obj_path}"
            )

            logging.info("=" * 80)
            logging.info("DATA TRANSFORMATION COMPLETED")
            logging.info("=" * 80)

            return (
                train_df,
                test_df,
                preprocessor,
            )

        except Exception as e:

            logging.error(str(e))

            raise CustomException(e, sys)