import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras_tuner import BayesianOptimization
from keras.callbacks import Callback

image_height = 308
image_width = 620
num_channels = 1

def load_images(folder_path, name):
    images = []; labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(f'.{name}.png'):
            image_path = os.path.join(folder_path, filename)
            image = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(image_height, image_width))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0
            images.append(image)
            labels.append(os.path.basename(folder_path))
    return np.array(images), np.array(labels)

class PrintValidationMetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nval_loss: {logs['val_loss']}, val_accuracy: {logs['val_accuracy']}")

class PrintBestValidationAccuracyCallback(Callback):
    def __init__(self):
        super(PrintBestValidationAccuracyCallback, self).__init__()
        self.best_val_accuracy = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy is not None and val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy

    def on_train_end(self, logs=None):
        print(f'\nBest val_accuracy: {self.best_val_accuracy}')

def build_model(hp):
    model = Sequential()
    model.add(Conv2D(filters=hp.Int('conv_filters', min_value=32, max_value=128, step=16),
                     kernel_size=hp.Choice('kernel_size', values=[3, 5]),
                     activation='relu',
                     input_shape=(image_height, image_width, num_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    for i in range(hp.Int('num_conv_layers', 1, 3)):
        model.add(Conv2D(filters=hp.Int(f'conv_{i}_filters', min_value=32, max_value=128, step=32),
                         kernel_size=hp.Choice(f'conv_{i}_kernel_size', values=[3, 5, 7]),
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    for i in range(hp.Int('num_dense_layers', 1, 2)):
        model.add(Dense(units=hp.Int(f'dense_{i}_units', min_value=32, max_value=128, step=32),
                        activation='relu'))
        model.add(Dropout(rate=hp.Float(f'dropout_{i}_rate', min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-5, 1e-4, 1e-3])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def handle(name):
        positives_images, positives_labels = load_images('/kaggle/input/positive-images', name)
        negatives_images, negatives_labels = load_images('/kaggle/input/negative-images', name)

        positives_labels = np.ones(len(positives_images))
        negatives_labels = np.zeros(len(negatives_images))

        indices = np.random.permutation(len(positives_images))
        positives_images = positives_images[indices]
        indices = np.random.permutation(len(negatives_images))
        negatives_images = negatives_images[indices]
        
        train_pos_images, rest_pos_images, train_pos_labels, rest_pos_labels = train_test_split(positives_images, positives_labels, test_size=0.2, random_state=42)
        train_neg_images, rest_neg_images, train_neg_labels, rest_neg_labels = train_test_split(negatives_images, negatives_labels, test_size=0.2, random_state=42)

        val_pos_images, test_pos_images, val_pos_labels, test_pos_labels = train_test_split(rest_pos_images, rest_pos_labels, test_size=0.5, random_state=42)
        val_neg_images, test_neg_images, val_neg_labels, test_neg_labels = train_test_split(rest_neg_images, rest_neg_labels, test_size=0.5, random_state=42)

        train_images = np.concatenate((train_pos_images, train_neg_images), axis=0)
        train_labels = np.concatenate((train_pos_labels, train_neg_labels), axis=0)

        val_images = np.concatenate((val_pos_images, val_neg_images), axis=0)
        val_labels = np.concatenate((val_pos_labels, val_neg_labels), axis=0)

        test_images = np.concatenate((test_pos_images, test_neg_images), axis=0)
        test_labels = np.concatenate((test_pos_labels, test_neg_labels), axis=0)

        tuner = BayesianOptimization(
            build_model,
            objective='val_accuracy',
            max_trials=10,
            directory=f'training/{name}',
            project_name='cnn_tuning'
        )

        callbacks = [PrintValidationMetricsCallback(), PrintBestValidationAccuracyCallback()]
        tuner.search(train_images, train_labels, epochs=10,  validation_data=(val_images, val_labels), callbacks=callbacks)

        best_model = tuner.get_best_models(num_models=1)[0]
        # best_model.fit(train_images, train_labels, epochs=10)
        best_model.save(f'best_model_of_{name}.h5')

        loaded_model = load_model(f'best_model_of_{name}.h5')
        test_loss, test_accuracy = best_model.evaluate(test_images, test_labels)
        print(f'Test Accuracy of {name}:', test_accuracy)
        print(f'Test Loss of {name}:', test_loss)

def main():
    handle('mean')
#     handle('median')
#     handle('iqr')

if __name__ == '__main__':
    main()