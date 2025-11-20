
"""
Add new or variations of models here as funcs.
"""

from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, GaussianNoise, MultiHeadAttention, Reshape)
from tensorflow.keras.optimizers import Adam

def create_xception_model(input_shape, num_classes=8, learning_rate=1e-4):
    inputs = Input(shape=input_shape, name="Input_Layer")
    base_model = Xception(weights="imagenet", input_tensor=inputs, include_top=False)
    base_model.trainable = False
    x = base_model.output
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]
    x = Reshape((height * width, channels), name="Reshape_to_Sequence")(x)
    x = MultiHeadAttention(num_heads=8, key_dim=channels, name="Multi_Head_Attention")(x, x)
    x = Reshape((height, width, channels), name="Reshape_to_Spatial")(x)
    x = GaussianNoise(0.25, name="Gaussian_Noise")(x)
    x = GlobalAveragePooling2D(name="Global_Avg_Pooling")(x)
    x = Dense(512, activation="relu", name="FC_512")(x)
    x = BatchNormalization(name="Batch_Normalization")(x)
    x = Dropout(0.25, name="Dropout")(x)
    outputs = Dense(num_classes, activation="softmax",name="Output_Layer")(x)
    model = Model(inputs=inputs, outputs=outputs, name="Xception_with_Attention")
    model.compile(
    optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def get_model(input, input_shape):
    model_options = {
        "1": create_xception_model,
        #TODO Add more models here
    }

    if input not in model_options:
        return None
    return model_options[input](input_shape)