from tensorflow.keras.applications import EfficientNetB0, ResNet50, DenseNet121, EfficientNetB4

# model = DenseNet121(weights="imagenet", include_top=False)
# model = EfficientNetB0(weights="imagenet", include_top=False)
model = EfficientNetB4(weights="imagenet", include_top=False)
# model = ResNet50(weights="imagenet", include_top=False)
model.save_weights("pretrained_weights/effnetb4_weights.weights.h5")