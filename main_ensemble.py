from models import get_model
from data_pipeline import Preprocessor
from train_pipeline import TrainPipeline
import tensorflow
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

def main(): # Test locally with `python main_ensemble.py`
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

    data_path = "."
    categories = ["Normal","Osteopenia", "Osteoporosis"]
    img_size = (224, 224)
    batch_size = 16
    epochs = 100
    
    preprocessor = Preprocessor(data_path, categories)
    print("Loading and preparing data...")
    train_gen, valid_gen, test_gen = preprocessor.preprocess(img_size=img_size, batch_size=batch_size)

    model_ids = ["2", "4", "5"] #TODO Update list here based on models to ensemble
    models = []
    predictions = []
    for model_id in model_ids:
        model = get_model(model_id, img_size + (3,))
        if model is None:
            print(f"Please provide a valid model input argument for model ID {model_id}.")
            continue
        
        print(f"Training model ID {model_id}...")
        trainer = TrainPipeline(model)
        trainer.train(train_gen, valid_gen, epochs=epochs)
        
        print(f"Predicting with model ID {model_id}...")
        pred = model.predict(test_gen)
        predictions.append(pred)
        models.append(model)
    
    preds = np.mean(predictions, axis=0)
    final_predicted_classes = np.argmax(preds, axis=1)
    test_labels = test_gen.classes

    print("Evaluating the model...")
    report = classification_report(test_labels, final_predicted_classes,
        target_names=list(test_gen.class_indices.keys()))
    print(report)

    conf_matrix = confusion_matrix(test_labels, final_predicted_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
    xticklabels=list(test_gen.class_indices.keys()),
    yticklabels=list(test_gen.class_indices.keys()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show(block=False)
    plt.pause(5)
    plt.close()

if __name__ == "__main__":
    main()