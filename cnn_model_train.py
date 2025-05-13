import numpy as np
import pickle
import cv2, os
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from keras import backend as K

K.set_image_data_format('channels_last')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
    for folder in os.listdir('gestures'):
        folder_path = os.path.join('gestures', folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.jpg'):
                    img_path = os.path.join(folder_path, file)
                    img = cv2.imread(img_path, 0)
                    if img is not None:
                        return img.shape
    raise ValueError("No valid image found in gestures/ directory.")

def get_num_of_classes():
    return len([folder for folder in os.listdir('gestures') if os.path.isdir(os.path.join('gestures', folder))])

image_x, image_y = get_image_size()

def cnn_model():
    num_classes = get_num_of_classes()
    model = Sequential()
    model.add(Conv2D(16, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    sgd = optimizers.SGD(learning_rate=0.005, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    checkpoint = ModelCheckpoint("cnn_model_keras2.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    return model, [checkpoint]

def train():
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f))

    with open("val_images", "rb") as f:
        val_images = np.array(pickle.load(f))
    with open("val_labels", "rb") as f:
        val_labels = np.array(pickle.load(f))

    if len(train_images) == 0 or len(train_labels) == 0:
        raise ValueError("Train data is empty!")
    if len(val_images) == 0 or len(val_labels) == 0:
        raise ValueError("Validation data is empty!")

    train_images, train_labels = shuffle(train_images, train_labels)
    val_images, val_labels = shuffle(val_images, val_labels)

    train_images = train_images.reshape(-1, image_x, image_y, 1).astype('float32') / 255
    val_images = val_images.reshape(-1, image_x, image_y, 1).astype('float32') / 255

    label_encoder = LabelEncoder()
    all_labels = np.concatenate([train_labels, val_labels])
    label_encoder.fit(all_labels)
    train_labels = label_encoder.transform(train_labels)
    val_labels = label_encoder.transform(val_labels)

    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)

    print(f"Training on {train_images.shape[0]} samples, {train_labels.shape[1]} classes.")

    model, callbacks = cnn_model()
    model.summary()
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels),
              epochs=20, batch_size=128, callbacks=callbacks)

    scores = model.evaluate(val_images, val_labels, verbose=0)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

train()
K.clear_session()
