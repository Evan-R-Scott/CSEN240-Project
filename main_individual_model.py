from models import get_model
from data_pipeline import Preprocessor
from train_pipeline import TrainPipeline
import sys
import tensorflow

def main(): # Test locally with `python main_individual_model.py <model_input_number>`
    print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

    data_path = "."
    categories = ["Normal","Osteopenia", "Osteoporosis"]
    img_size = (224, 224)
    batch_size = 32
    epochs = 100

    if len(sys.argv) > 1:
        model_input = sys.argv[1]
    else:
        sys.exit("Please provide a model input argument.")

    model = get_model(model_input, img_size + (3,))
    if model is None:
        sys.exit("Please provide a valid model input argument.")
    
    preprocessor = Preprocessor(data_path, categories)
    print("Loading and preparing data...")
    train_gen, valid_gen, test_gen = preprocessor.preprocess(img_size=img_size, batch_size=batch_size)

    print("Training the model...")
    trainer = TrainPipeline(model)
    trainer.train(train_gen, valid_gen, epochs=epochs)

    print("Evaluating the model...")
    trainer.evaluate(test_gen)

if __name__ == "__main__":
    main()