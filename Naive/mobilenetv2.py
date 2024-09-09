import os
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Reshape, Dropout, Conv2D, Activation
from keras.applications import MobileNet
import cv2
import numpy as np
import time

def emotion_recognition_model(weight_path):
    base_model = MobileNet(weights=None, include_top=False, input_shape=(128, 128, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(1e-3, name='dropout')(x)
    x = Conv2D(7, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    predictions = Reshape((7,), name='reshape_2')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(weight_path)
    return model

def process_frame(img, model, labels):
    face_input = np.zeros((1, 128, 128, 3))
    face_img = cv2.resize(img, (128, 128))[:,:,::-1] / 255.0
    face_input[0] = face_img
    pred_prob = model.predict(face_input)
    pred_label = labels[np.argmax(pred_prob[0])]
    return pred_label

def main():
    WEIGHTS_PATH = "mobilenet_rafdb.hdf5"
    er_model = emotion_recognition_model(WEIGHTS_PATH)
    labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
    
    cap = cv2.VideoCapture(0)
    t_start = time.time()
    fps = 0
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        pred_label = process_frame(img, er_model, labels)
        
        # Display the prediction on the image
        cv2.putText(img, pred_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        fps = fps + 1
        sfps = fps / (time.time() - t_start)
        cv2.putText(img, "FPS: " + str(int(sfps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow('Emotion Recognition', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()