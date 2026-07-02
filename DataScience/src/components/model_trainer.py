import os
import sys
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
dagshub.init(repo_owner='Immanuel2004', repo_name='Ml-Classification', mlflow=True)

@dataclass
class ModelTrainerConfig:
    model_dir: str = os.path.join("artifacts", "models")
    best_model_path: str = os.path.join("artifacts", "models", "best_model.pkl")
    model_report_path: str = os.path.join("artifacts", "models", "model_report.csv")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        os.makedirs(self.model_trainer_config.model_dir, exist_ok=True)

        mlflow.set_experiment("Classification-Experiment")

    def evaluate_model(self, model, X_train, y_train, X_test, y_test):
        """Train and evaluate a regression model."""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        try:

            if len(np.unique(y_test)) == 2:

                if hasattr(model, "predict_proba"):

                    y_prob = model.predict_proba(X_test)[:, 1]

                    auc = roc_auc_score(y_test, y_prob)

                    metrics["roc_auc"] = auc

        except Exception:
            pass

        return metrics
    

    def log_to_mlflow(self, model_name, model, params, metrics):
        """Log params, metrics to MLflow (DagsHub) and save model locally."""
        with mlflow.start_run(run_name=model_name):
            if params:
                mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                sk_model=model,
                name=model_name,
                serialization_format="cloudpickle",
            )
            joblib.dump(
                model,
                os.path.join(
                    self.model_trainer_config.model_dir,
                    f"{model_name}.pkl",
                ),
            )

            logging.info(f"{model_name} saved successfully.")
    def initiate_model_trainer(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str = None,
        tune_hyperparameters: bool = True,
    ):
        
        try:
            logging.info("Model Training initiated..")
            if target_column is None:
                target_column = train_df.columns[-1]

            logging.info(f"Target Column : {target_column}")
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            models = {

                "LogisticRegression": LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),

                "DecisionTree": DecisionTreeClassifier(
                    random_state=42,
                ),

                "RandomForest": RandomForestClassifier(
                    random_state=42,
                ),

                "ExtraTrees": ExtraTreesClassifier(
                    random_state=42,
                ),

                "GradientBoosting": GradientBoostingClassifier(
                    random_state=42,
                ),

                "AdaBoost": AdaBoostClassifier(
                    random_state=42,
                ),

                "KNeighbors": KNeighborsClassifier(),

                "SVM": SVC(
                    probability=True,
                ),

                "XGBoost": XGBClassifier(
                    eval_metric="logloss",
                    random_state=42,
                ),

                "CatBoost": CatBoostClassifier(
                    verbose=0,
                    random_state=42,
                ),
            }
            params = {

                "LogisticRegression": {
                    "C": [0.1, 1]
                },

                "DecisionTree": {
                    "max_depth": [5, 10]
                },

                "RandomForest": {
                    "n_estimators": [100],
                    "max_depth": [10, None],
                },

                "ExtraTrees": {
                    "n_estimators": [100],
                    "max_depth": [10, None],
                },

                "GradientBoosting": {
                    "n_estimators": [100],
                    "learning_rate": [0.1],
                },

                "AdaBoost": {
                    "n_estimators": [100],
                    "learning_rate": [0.1],
                },

                "KNeighbors": {
                    "n_neighbors": [5, 7]
                },

                "SVM": {
                    "C": [1],
                    "kernel": ["rbf"],
                },

                "XGBoost": {
                    "n_estimators": [100],
                    "learning_rate": [0.1],
                    "max_depth": [5],
                },

                "CatBoost": {
                    "iterations": [200],
                    "depth": [6],
                    "learning_rate": [0.1],
                },
            }


            model_report = {}
            best_model = None
            best_model_name = None
            best_accuracy = -np.inf
            
            for name, model in models.items():

                logging.info(f"Training {name}")
                best_params = None
                if tune_hyperparameters and name in params:
                    grid = GridSearchCV(
                        estimator=model,
                        param_grid=params[name],
                        scoring="accuracy",
                        cv=2,
                        n_jobs=-1,
                    )
                    grid.fit(X_train, y_train)
                    best_model_instance = grid.best_estimator_
                    best_params = grid.best_params_
                    logging.info(best_params)

                else:

                    best_model_instance = model
                    best_model_instance.fit(
                        X_train,
                        y_train,
                    )

                metrics = self.evaluate_model(
                    best_model_instance,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                )

                self.log_to_mlflow(
                    name,
                    best_model_instance,
                    best_params,
                    metrics,
                )

                model_report[name] = metrics

                if metrics["accuracy"] > best_accuracy:

                    best_accuracy = metrics["accuracy"]
                    best_model = best_model_instance
                    best_model_name = name

            report_df = pd.DataFrame(model_report).T
            report_df.to_csv(
                self.model_trainer_config.model_report_path,
                index=True,
            )
            joblib.dump(
                best_model,
                self.model_trainer_config.best_model_path,
            )
            logging.info(
                f"Best Model : {best_model_name}"
            )
            logging.info(
                f"Accuracy : {best_accuracy}"
            )

            return {

                "best_model_name": best_model_name,
                "best_accuracy": best_accuracy,
                "best_model": best_model,
                "model_report": report_df,
            }
        except Exception as e:
            raise CustomException(e, sys)