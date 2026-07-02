from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging


def main():
    try:

        logging.info("=" * 100)
        logging.info("ML CLASSIFICATION PIPELINE STARTED")
        logging.info("=" * 100)

        ingestion = DataIngestion()

        df, dataset_path = ingestion.initiate_data_ingestion()

        logging.info("Data Ingestion Completed")

        preprocessing = DataPreprocessing()

        cleaned_df = preprocessing.initiate_data_preprocessing(df)

        logging.info("Data Preprocessing Completed")

        transformation = DataTransformation()

        train_df, test_df, preprocessor = (
            transformation.initiate_data_transformation(cleaned_df)
        )

        logging.info("Data Transformation Completed")

        trainer = ModelTrainer()

        result = trainer.initiate_model_trainer(
            train_df=train_df,
            test_df=test_df,
            tune_hyperparameters=True,
        )

        logging.info("=" * 100)
        logging.info("MODEL TRAINING COMPLETED")
        logging.info("=" * 100)

        logging.info(
            f"Best Model : {result['best_model_name']}"
        )

        logging.info(
            f"Best Accuracy : {result['best_accuracy']:.4f}"
        )

        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETED")
        print("=" * 60)
        print(f"Best Model     : {result['best_model_name']}")
        print(f"Best Accuracy  : {result['best_accuracy']:.4f}")
        print("=" * 60)

    except Exception as e:

        logging.exception("Pipeline execution failed")

        print(f"\nError : {e}")


if __name__ == "__main__":
    main()