import tensorflow as tf
from Cnnclassifier.constants import VALIDATION_SPLIT
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from Cnnclassifier.utils.common import save_json
from Cnnclassifier.entity.config_entity import EvaluationConfig
class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=VALIDATION_SPLIT
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluate_model(self):
        self.model = self.load_model(self.config.path_model)
        self._valid_generator()
        # Fixed: using self.model and capturing the evaluation score
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        score = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=score)
        print(f"score : {score}")

    def log_into_mlflow(self):
        """Logs parameters, metrics, and the model to DagsHub/MLflow."""
        # 1. Force the remote tracking URI (prevents local mlruns folder)
        mlflow.set_tracking_uri(self.config.mlflow_url)
        
        # 2. Set an explicit experiment name 
        # (Avoids INVALID_PARAMETER_VALUE errors on the 'Default' experiment)
        mlflow.set_experiment("Chest_Cancer_Classification_VGG16")
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            # Log all parameters from params.yaml
            mlflow.log_params(self.config.all_params)
            
            # Log final metrics (loss and accuracy)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            
            # Log the local scores.json as an artifact
            mlflow.log_artifact("scores.json")
            
            # 3. Log the model as an artifact 
            # (Removing registered_model_name to avoid 403 Forbidden errors)
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model")
            else:
                mlflow.keras.log_model(self.model, "model")