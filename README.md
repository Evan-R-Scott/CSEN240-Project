### To Run in HPC Cluster

Doesn't download the modelfile so right now it is a run-and-done workflow. Not sure if we need to download our best model for submission?

1. ssh (your SCU username)@login.wave.scu.edu
2. git clone https://github.com/Evan-R-Scott/CSEN240-Project.git
3. cd CSEN240-Project/
4. sbatch slurm.sh

Check progress using commands like:
  1. squeue
  2. squeue -u (username)
  3. tail -f csen240_project.err
  4. tail -f csen240_project.log -> Best


Knee Osteoporosis Detection – ML Model

This project trains a deep learning model to classify knee bone density into three medical diagnostic categories:

Normal

Osteopenia

Osteoporosis

The project includes code for:
✔ dataset preprocessing
✔ oversampling for class balance
✔ model training
✔ model evaluation
✔ model persistence to disk
✔ single-image inference using a trained model

🧠 Model

We use an Xception CNN pretrained on ImageNet combined with:

Multi-Head Attention

GlobalAveragePooling

BatchNormalization

Dropout regularization

Defined in models.py inside create_xception_model().

📂 Directory Structure
train/
   Normal/
   Osteopenia/
   Osteoporosis/
data_pipeline.py
models.py
train_pipeline.py
main.py
predict.py
requirements.txt
README.md

🏋️‍♂️ Training the Model

Activate the virtual environment:

venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Train the model:

python main.py 1


This will:

✓ load and balance dataset
✓ train for 20 epochs
✓ evaluate on test data
✓ save the model to:

trained_model.keras

🔍 Predict on a Single Image

Once the model is trained:

python predict.py path_to_image.png


Example:

python predict.py test_images/arthiknee.png


Output:

Prediction: Osteopenia (confidence: 0.8421)

🧪 Data Handling (data_pipeline.py)

Includes:

directory scanning

label encoding

oversampling using RandomOverSampler

class balancing

train/val/test split

image generation using ImageDataGenerator

📊 Evaluation

During training, the following outputs are generated:

training/validation accuracy curves

training/validation loss curves

confusion matrix

full classification report (precision/recall/F1)

📦 Model Saving

At end of training:

self.model.save("trained_model.keras")


Saved model can be reused for inference without retraining.

✨ Planned Future Enhancements

Grad-CAM visualization

Streamlit web UI for uploading knee scans

Support for additional model architectures

Automated dataset augmentation

Export to ONNX / TensorRT for real-time inference

👨‍💻 Contributors

Evan and Abhilash