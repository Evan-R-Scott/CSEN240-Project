
"""
Add new or variations of models here as funcs.
"""

from tensorflow.keras.applications import Xception, ResNet50, DenseNet121, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, GaussianNoise, MultiHeadAttention, Reshape)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalFocalCrossentropy

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

def create_resnet_model(input_shape, num_classes=3, learning_rate=1e-4):
    inputs = Input(shape=input_shape)
    base_model = ResNet50(
        weights=None,
        include_top=False,
        input_tensor=inputs
    )

    base_model.load_weights("pretrained_weights/resnet50_weights.weights.h5")
    
    num_layers = len(base_model.layers)
    for layer in base_model.layers[:-int(num_layers * 0.05)]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(512, activation="relu", kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)

    x = Dense(256, activation="relu", kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation="relu", kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    focal_loss = CategoricalFocalCrossentropy(gamma=2.0, alpha=0.25)

    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    # model.compile(
    #     optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss=focal_loss, metrics=["accuracy"]
    )
    
    return model

def create_vit_model(input_shape, num_classes=3, learning_rate=5e-5):
    import keras_hub

    vit = keras_hub.models.ViTBackbone.from_preset("vit_large_patch16_224_imagenet")
    # vit_base_patch16_224_imagenet
    # vit.trainable = False

    for i, layer in enumerate(vit.layers):
        if i < len(vit.layers) * 0.5:
            layer.trainable = False

    inputs = Input(shape=input_shape)
    x = vit(inputs)
    x = x[:, 0, :]

    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)

    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def create_densenet_model(input_shape, num_classes=3, learning_rate=1e-4):
    inputs = Input(shape=input_shape)
    base_model = DenseNet121(weights=None, include_top=False, input_tensor=inputs)
    base_model.load_weights("pretrained_weights/densenet121_weights.weights.h5")

    num_layers = len(base_model.layers)
    for layer in base_model.layers[:-int(num_layers * 0.05)]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(256, activation="relu", kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation="relu", kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # x = Dense(64, activation="relu", kernel_regularizer=l2(5e-4))(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    focal_loss = CategoricalFocalCrossentropy(gamma=2.0, alpha=0.25)

    model = Model(inputs=inputs, outputs=outputs)
    # model.compile(
    #     optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    # )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss=focal_loss, metrics=["accuracy"]
    )
    return model

def create_efficientnet_model(input_shape, num_classes=3, learning_rate=1e-4):

    inputs = Input(shape=input_shape)
    base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=inputs)
    base_model.load_weights("pretrained_weights/efficientnetb0_weights.weights.h5")

    num_layers = len(base_model.layers)
    for layer in base_model.layers[:-int(num_layers * 0.05)]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(256, activation="relu", kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation="relu", kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # x = Dense(64, activation="relu", kernel_regularizer=l2(5e-4))(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    focal_loss = CategoricalFocalCrossentropy(gamma=2.0, alpha=0.25)

    model = Model(inputs=inputs, outputs=outputs)
    # model.compile(
    #     optimizer=Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    # )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss=focal_loss, metrics=["accuracy"]
    )
    return model

def get_model(input, input_shape):
    model_options = {
        "1": create_xception_model,
        "2": create_resnet_model, 
        "3": create_vit_model, 
        "4": create_densenet_model,
        "5": create_efficientnet_model,
        #TODO List new or variations of models here
    }

    if input not in model_options:
        return None
    return model_options[input](input_shape)