
"""
Add new or variations of models here as funcs.
"""

from tensorflow.keras.applications import Xception, ResNet50
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

def create_dense_model(input_shape, num_classes=8, learning_rate=1e-4):
    backbone = ResNet50(
        weights=None,
        include_top=False,
        input_shape=input_shape,
    )

    backbone.load_weights("resnet50_weights.weights.h5")
    
    for layer in backbone.layers[:-20]:
        layer.trainable = False

    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=backbone.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    return model

def create_vit_model(input_shape, num_classes=8, learning_rate=1e-4):
    import keras_hub
    
    vit = keras_hub.models.ViTBackbone.from_preset("vit_base_patch16_224_imagenet")
    vit.trainable = False

    inputs = Input(shape=input_shape)
    x = vit(inputs)
    x = x[:, 0, :]

    x = Dense(512, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def get_model(input, input_shape):
    model_options = {
        "1": create_xception_model,
        "2": create_dense_model, 
        "3": create_vit_model, 
        #TODO List new or variations of models here
    }

    if input not in model_options:
        return None
    return model_options[input](input_shape)