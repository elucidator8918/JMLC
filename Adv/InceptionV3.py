# -*- coding: utf-8 -*-

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def create_model(num_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3))
    
    # Fine-tune from this layer onwards
    fine_tune_at = 25
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False       
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)    
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='leaky_relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def get_callbacks(model_name):
    callbacks = []
    
    checkpoint = ModelCheckpoint(
        filepath=f'model.{model_name}.keras',
        monitor='accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint)
    train_generator.reset()
    val_generator.reset()
    return callbacks

# Hyperparameters
epochs = 250
batch_size = 128
initial_learning_rate = 1e-3
targetx, targety = 75, 75
classes = 7
testsplit = 0.2
seed = random.randint(1, 1000)
print(f"Random seed: {seed}")

target_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

data_dir = "FER2013/train"
data_dir1 = "FER2013/test"

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    fill_mode='nearest'
)

# Validation data should only be rescaled
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(targetx, targety),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=seed,
    
)

val_generator = val_datagen.flow_from_directory(
    data_dir1,
    target_size=(targetx, targety),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=seed
)

# Create and compile the model
model = create_model(classes)
optimizer = AdamW(learning_rate=initial_learning_rate, weight_decay=1e-2)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'AUC', 'F1Score'])
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=epochs,
    callbacks=get_callbacks('inception_finetuned')
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

predictions = model.predict(val_generator, steps=len(val_generator))
y = np.argmax(predictions, axis=1)

print('Classification Report')
cr = classification_report(y_true=val_generator.classes, y_pred=y, target_names=val_generator.class_indices)
print(cr)

Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true=val_generator.classes
cm=confusion_matrix(y_true,y_pred)
cm

f, ax=plt.subplots(figsize=(12,8))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax,xticklabels=target_names, yticklabels=target_names,)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

row_sums = cm.sum(axis=1, keepdims=True)
cm_percentage = (cm / row_sums) * 100
f, ax = plt.subplots(figsize=(12, 8))
plt.xlabel("y_pred")
plt.ylabel("y_true")
sns.heatmap(cm_percentage, annot=True, linewidths=0.5, linecolor="red", fmt=".1f", ax=ax, xticklabels=target_names, yticklabels=target_names)
plt.show()

model.evaluate(val_generator , verbose = 1)