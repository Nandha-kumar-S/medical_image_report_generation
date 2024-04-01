import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class ImageEncoder:
    def __init__(self,dataset, image_dir, target_size=(224, 224), batch_size=32):
        self.image_dir = image_dir
        self.dataset = dataset
        self.target_size = target_size
        self.batch_size = batch_size
        self.base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        self.datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    def extract_image_features(self):
        frontal_generator = self.datagen.flow_from_dataframe(
            dataframe=self.dataset,
            directory=self.image_dir,
            x_col='Frontal',
            y_col=None,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False
        )

        lateral_generator = self.datagen.flow_from_dataframe(
            dataframe=self.dataset,
            directory=self.image_dir,
            x_col='Lateral',
            y_col=None,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False
        )

        frontal_features = self.base_model.predict(frontal_generator)
        lateral_features = self.base_model.predict(lateral_generator)

        return frontal_features, lateral_features
    