import os
import sys
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from src.logger import *
from src.exception import *
from src.entity.config_entity import *

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        # Using InceptionV3 with custom input shape, pre-trained on ImageNet
        self.model = tf.keras.applications.InceptionV3(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Flatten the output of the base model
        flatten_in = tf.keras.layers.Flatten()(model.output)

        # Add dropout to reduce overfitting
        dropout = tf.keras.layers.Dropout(0.5)(flatten_in)

        # Dense layer with L2 regularization
        regularizer = tf.keras.regularizers.l2(0.01)
        dense_out = tf.keras.layers.Dense(
            units=classes,
            activation="softmax",
            kernel_regularizer=regularizer
        )(dropout)

        # Full model with base model and added layers
        full_model = tf.keras.models.Model(inputs=model.input, outputs=dense_out)

        # Compile the model with a learning rate scheduler
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)