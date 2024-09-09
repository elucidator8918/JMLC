## RUN ON KAGGLE
## !pip install tensorflow==2.11 transformers datasets tensorboard -U -q

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Permute
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from transformers import TFViTModel
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
CONFIG = {
    'BATCH_SIZE': 32,
    'IMG_SIZE': 256,
    'NUM_CLASSES': 7,
    'EPOCHS': 50,
    'LEARNING_RATE': 1e-5,
}

CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Data paths
TRAIN_DIR = "FER2013/train"
TEST_DIR = "FER2013/test"

def load_datasets(train_dir, test_dir, img_size, batch_size):
    """Load and prepare datasets."""
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="training"
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="validation"
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_NAMES,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=False,
    )

    return train_dataset, val_dataset, test_dataset

def create_model(img_size, num_classes):
    """Create and return the ViT model."""
    resize_rescale_permute = tf.keras.Sequential([
        tf.keras.layers.Resizing(224, 224),
        tf.keras.layers.Rescaling(1./255),
        Permute((3,1,2))
    ])

    base_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    inputs = Input(shape=(img_size, img_size, 3))
    x = resize_rescale_permute(inputs)
    x = base_model.vit(x)[0][:,0,:]
    x = Dropout(0.3)(x)
    output = Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    
    return Model(inputs=inputs, outputs=output)

class F1Score(tf.keras.metrics.Metric):
    """Custom F1 Score metric."""
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = Precision(thresholds=0.5)
        self.recall_fn = Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_states(self):
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)

def train_model(model, train_dataset, val_dataset, epochs):
    """Train the model and return training history."""
    model.compile(
        optimizer=AdamW(learning_rate=CONFIG['LEARNING_RATE']),
        loss=CategoricalCrossentropy(from_logits=True),
        metrics=[CategoricalAccuracy(), F1Score()]
    )

    checkpoint = ModelCheckpoint(
        filepath='model_{epoch}.keras',
        save_best_only=False
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        verbose=1,
        restore_best_weights=True
    )
    csv_logger = CSVLogger("training_log.csv")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, csv_logger]
    )

    return history

def plot_training_history(history):
    """Plot training history."""
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Time')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def evaluate_model(model, test_dataset):
    """Evaluate the model on test data and print metrics."""
    actual_labels = []
    predicted_labels = []

    for images, labels in test_dataset:
        preds = model.predict(images)
        pred_labels = np.argmax(preds, axis=1)
        actual_labels.extend(np.argmax(labels.numpy(), axis=1))
        predicted_labels.extend(pred_labels)

    actual_labels = np.array(actual_labels)
    predicted_labels = np.array(predicted_labels)

    print(classification_report(actual_labels, predicted_labels, target_names=CLASS_NAMES))

    cm = confusion_matrix(actual_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    train_dataset, val_dataset, test_dataset = load_datasets(
        TRAIN_DIR, TEST_DIR, CONFIG['IMG_SIZE'], CONFIG['BATCH_SIZE']
    )

    model = create_model(CONFIG['IMG_SIZE'], CONFIG['NUM_CLASSES'])
    history = train_model(model, train_dataset, val_dataset, CONFIG['EPOCHS'])
    
    plot_training_history(history)
    evaluate_model(model, test_dataset)

    model.save('vit_model.keras')

if __name__ == "__main__":
    main()