import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataPreprocessingConfig:
    processed_data_dir = os.path.join("artifacts", "processed_data")

    cleaned_data_path = os.path.join(
        processed_data_dir,
        "cleaned_data.csv",
    )

    summary_statistics_path = os.path.join(
        processed_data_dir,
        "summary_statistics.csv",
    )

    missing_values_path = os.path.join(
        processed_data_dir,
        "missing_values.csv",
    )

    correlation_plot_path = os.path.join(
        processed_data_dir,
        "correlation_heatmap.png",
    )

    class_distribution_path = os.path.join(
        processed_data_dir,
        "class_distribution.png",
    )


class DataPreprocessing:

    def __init__(self):
        self.preprocessing_config = DataPreprocessingConfig()

    def initiate_data_preprocessing(
        self,
        df: pd.DataFrame,
        target_column: str = None,
    ):

        try:

            logging.info("=" * 80)
            logging.info("DATA PREPROCESSING STARTED")
            logging.info("=" * 80)

            df = df.copy()

            if target_column is None:
                target_column = df.columns[-1]

            if target_column not in df.columns:
                raise ValueError(
                    f"Target column '{target_column}' not found."
                )

            logging.info(f"Target Column : {target_column}")

            os.makedirs(
                self.preprocessing_config.processed_data_dir,
                exist_ok=True,
            )

            logging.info(f"Dataset Shape : {df.shape}")
            logging.info(f"Columns : {list(df.columns)}")

            desc = df.describe(include="all")

            desc.to_csv(
                self.preprocessing_config.summary_statistics_path
            )

            logging.info("Summary statistics saved.")

            missing = df.isnull().sum()

            missing.to_csv(
                self.preprocessing_config.missing_values_path
            )

            logging.info(f"\nMissing Values\n{missing}")

            duplicate_count = df.duplicated().sum()

            logging.info(f"Duplicate Rows : {duplicate_count}")

            if duplicate_count > 0:

                df = df.drop_duplicates(ignore_index=True)

                logging.info("Duplicate rows removed.")

            for col in df.columns:

                if pd.api.types.is_numeric_dtype(df[col]):

                    df[col] = df[col].fillna(df[col].mean())

                else:

                    mode = df[col].mode()

                    if not mode.empty:
                        df[col] = df[col].fillna(mode.iloc[0])

            logging.info("Missing values handled.")

            constant_cols = [
                col
                for col in df.columns
                if df[col].nunique() == 1
            ]

            if constant_cols:

                logging.info(
                    f"Removing constant columns : {constant_cols}"
                )

                df.drop(
                    columns=constant_cols,
                    inplace=True,
                )

            numeric_cols = df.select_dtypes(
                include=[np.number]
            ).columns

            logging.info("=" * 80)
            logging.info("OUTLIER DETECTION")
            logging.info("=" * 80)

            for col in numeric_cols:

                q1 = df[col].quantile(0.25)

                q3 = df[col].quantile(0.75)

                iqr = q3 - q1

                lower = q1 - (1.5 * iqr)

                upper = q3 + (1.5 * iqr)

                outliers = df[
                    (df[col] < lower)
                    | (df[col] > upper)
                ]

                logging.info(
                    f"{col} : {len(outliers)} outliers"
                )

            corr = df.corr(numeric_only=True)

            plt.figure(figsize=(12, 8))

            sns.heatmap(
                corr,
                cmap="coolwarm",
                annot=False,
                linewidths=0.5,
            )

            plt.title("Correlation Heatmap")

            plt.tight_layout()

            plt.savefig(
                self.preprocessing_config.correlation_plot_path
            )

            plt.close()

            logging.info("Correlation heatmap saved.")

            logging.info("=" * 80)
            logging.info("CLASS DISTRIBUTION")
            logging.info("=" * 80)

            logging.info(
                f"\n{df[target_column].value_counts()}"
            )

            plt.figure(figsize=(8, 5))

            sns.countplot(
                data=df,
                x=target_column,
            )

            plt.title("Target Class Distribution")

            plt.tight_layout()

            plt.savefig(
                self.preprocessing_config.class_distribution_path
            )

            plt.close()

            logging.info("Class distribution plot saved.")

            df.to_csv(
                self.preprocessing_config.cleaned_data_path,
                index=False,
            )

            logging.info(
                f"Cleaned dataset saved to : "
                f"{self.preprocessing_config.cleaned_data_path}"
            )

            logging.info("=" * 80)
            logging.info("DATA PREPROCESSING COMPLETED")
            logging.info("=" * 80)

            return df

        except Exception as e:

            logging.error(
                f"Error during preprocessing : {str(e)}"
            )

            raise CustomException(e, sys)