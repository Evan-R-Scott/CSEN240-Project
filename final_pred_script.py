import sys
import numpy as np
from sklearn.metrics import classification_report
from models import get_model
from data_pipeline import Preprocessor

def predict(model, data, dataset_name):
    labels = data.classes
    predictions = model.predict(data)
    predicted_classes = np.argmax(predictions, axis=1)

    print(f"Evaluation Report for {dataset_name} Dataset:\n")
    report = classification_report(labels, predicted_classes,
        target_names=list(data.class_indices.keys()))
    print(report)

def main(): # Test locally with `python final_pred_script.py <model_input_number>`

    data_path = "preprocessed_data"
    categories = ["Normal","Osteopenia", "Osteoporosis"]
    img_size = (224, 224)
    batch_size = 32

    if len(sys.argv) > 1:
        model_input = sys.argv[1]
    else:
        sys.exit("Please provide a model input argument.")

    model = get_model(model_input, img_size + (3,))
    if model is None:
        sys.exit("Please provide a valid model input argument.")
    
    model.load_weights("best_model_weights.weights.h5")

    preprocessor = Preprocessor(data_path, categories)
    train_gen, valid_gen, test_gen = preprocessor.preprocess(img_size=img_size, batch_size=batch_size)

    predict(model, train_gen, dataset_name="Train")
    predict(model, valid_gen, dataset_name="Validation")
    predict(model, test_gen, dataset_name="Test")

if __name__ == "__main__":
    main()