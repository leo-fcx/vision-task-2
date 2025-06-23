import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

DATA_DIR = os.path.abspath("./data/subclasses")  # Directorio con carpetas por subclase

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

def create_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    datagen = ImageDataGenerator(validation_split=0.2,
                                 rescale=1./255,
                                 horizontal_flip=True,
                                 zoom_range=0.2)

    train_gen = datagen.flow_from_directory(DATA_DIR,
                                            target_size=IMG_SIZE,
                                            batch_size=BATCH_SIZE,
                                            subset='training',
                                            class_mode='categorical')

    val_gen = datagen.flow_from_directory(DATA_DIR,
                                          target_size=IMG_SIZE,
                                          batch_size=BATCH_SIZE,
                                          subset='validation',
                                          class_mode='categorical')

    model = create_model()

    model.fit(train_gen,
              validation_data=val_gen,
              epochs=EPOCHS)

    model.save("classifier/model_subclases_cellphones.keras")
    print("Modelo guardado en classifier/model_subclases_cellphones.keras")

if __name__ == "__main__":
    train()
