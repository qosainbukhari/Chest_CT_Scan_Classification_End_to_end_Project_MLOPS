import tensorflow as tf
from Cnnclassifier.constants import VALIDATION_SPLIT
from pathlib import Path
from Cnnclassifier.entity.config_entity import ModelTrainingConfig


class Training:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None

    def get_base_model(self):
        # ✅ Load model WITHOUT optimizer state
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path,
            compile=False
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
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

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        return self.train_generator, self.valid_generator

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)

    def train(self):

        # ✅ Compile ONCE here
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        # ✅ DO NOT pass steps_per_epoch
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            callbacks=[early_stopping]
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )