from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight


class TrainPipeline:
    def __init__(self, model):
        self.model = model
    
    def train(self, train_gen, valid_gen, epochs=20):
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=25,
            restore_best_weights=True, mode='max', verbose=1)
        
        lr_handler = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
            patience=8, min_lr=1e-7, verbose=1)

        history = self.model.fit( train_gen,
                    validation_data=valid_gen,
                    epochs=epochs,
                    callbacks=[early_stopping, lr_handler],
                    verbose=1)
        # y_pred = self.model.predict(valid_gen)
        # y_true = valid_gen.labels

        # ppo_loss_value = self.ppo_loss(y_true, y_pred)
        # print("\nPPO Loss on Validation Data:", ppo_loss_value.numpy())

        self.model.save_weights("best_model_weights.h5")

        self.plot_history(history)
    
    def evaluate(self, test_gen):
        test_labels = test_gen.classes
        predictions = self.model.predict(test_gen)
        predicted_classes = np.argmax(predictions, axis=1)

        self.plot_results(test_gen, test_labels, predicted_classes)

    def ppo_loss(self, y_true, y_pred):
        epsilon = 0.2
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        selected_probs = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1)
        old_selected_probs = tf.reduce_sum(tf.stop_gradient(y_pred) * y_true_one_hot, axis=-1)
        ratio = selected_probs / (old_selected_probs + 1e-10)
        clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
        loss = -tf.reduce_mean(tf.minimum(ratio, clipped_ratio))
        return loss

    def plot_history(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        #plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show(block=False)
        plt.pause(5)
        plt.close()
    
    def plot_results(self, test_gen, test_labels, predicted_classes):
        report = classification_report(test_labels, predicted_classes,
        target_names=list(test_gen.class_indices.keys()))
        print(report)

        conf_matrix = confusion_matrix(test_labels, predicted_classes)

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
