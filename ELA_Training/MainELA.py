import os 
import cv2
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

'''
Does ELA on a given image by calculating abs_diff and 
scaling(dynamically brightening, avg extremas)
'''
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = sum([ex[1] for ex in extrema])/3
    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    return ela_image

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize((128,128))).flatten() / 255.0

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def scheduler(epoch):
    lr = 1e-4
    if epoch < 10:
      return lr
    else:
      return lr * (0.9)

def main():
    # This script is intended for training the model.
    # It requires the CASIA2 dataset to be present in the 'Casia2' directory.
    
    # Use os.path.join for Windows/Linux cross-compatibility
    Real_path = os.path.join('Casia2', 'Real')
    Tampered_path = os.path.join('Casia2', 'Tampered')
    
    if not os.path.exists(Real_path) or not os.path.exists(Tampered_path):
        print("CASIA2 dataset not found. Skipping training part.")
        print("Please place the dataset in 'Casia2/Real' and 'Casia2/Tampered' to run training.")
        print("You can download the CASIA2 dataset from: https://www.kaggle.com/datasets/sophiasophia/casia-20-image-tampering-detection-dataset")
        return

    X = []
    Y = []
    paths = [Real_path, Tampered_path] 
    label = 0
    for i in paths:
        print(f"Processing {i} Images:")
        for name in os.listdir(i):
            img_path = os.path.join(i,name)
            if img_path.lower().endswith(('.jpeg', '.jpg')):
                img = prepare_image(image_path=img_path)
                X.append(img)
                Y.append(label)
                if len(Y) % 1000 == 0: 
                    print(f"\tProcessed {len(Y)} images")
        label = 1

    X = np.array(X).astype("float64")
    Y = tf.keras.utils.to_categorical(Y, 2)
    X = X.reshape(-1, 128, 128, 3)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=5)

    base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    output = Dense(2, activation='softmax')(x)
    
    model = Model(base_model.inputs, output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=1e-4, momentum=0.95, nesterov=False),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    reduce_lr = LearningRateScheduler(scheduler)
    
    print("Starting training...")
    history = model.fit(
        X_train, Y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, Y_val),
        callbacks=[reduce_lr]
    )

    model.save('model_ela_new.h5')
    print("Model saved as model_ela_new.h5")

if __name__ == "__main__":
    main()
