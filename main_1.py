
import glob
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import  Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.layers import  Conv2D, Dense, Dropout, Flatten, MaxPool2D 

def obtain_images_and_labels():
    images=[]
    for i in glob.glob('files/*.jpg'):
        image=cv.imread(i,1)
        images.append(image)
    images=np.asanyarray(images)
    labels=np.genfromtxt('GENKI-4K_Labels.txt', delimiter=" ")[:,0]
    return images,labels

def obtain_model():
    ### standard model for images and binary classification with loss='binary_crossentropy'
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=5, padding="same", activation="relu",
                     input_shape= (64,64,3)))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(filters=32, kernel_size=4, padding="same", activation="relu"))
    model.add(Dropout(0.2))
    model.add(MaxPool2D(2,2))
    model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(Dropout(0.2))
    model.add(MaxPool2D(2,2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def obtain_faces(images):
    ### crop images
    cascPath =cv.CascadeClassifier('faces_classifier.xml')
    list_of_faces=[]
    for i, image in enumerate(images):
        gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces=cascPath.detectMultiScale(gray, 1.3, 5)
        if len(faces)>0:
            (x,y,w,h)=faces[0]
            roi=image[y:y+h,x:x+w]
            resized_image=cv.resize(roi,(64,64),interpolation=cv.INTER_CUBIC)
            list_of_faces.append(resized_image)
        else:
            resized_image=cv.resize(image,(64,64),interpolation=cv.INTER_CUBIC)
            list_of_faces.append(resized_image)
    list_of_faces = np.asanyarray(list_of_faces)
    return list_of_faces

if __name__ == '__main__':
    ### get images and labels
    X,y=obtain_images_and_labels()
    ### crop images
    faces = obtain_faces(X)
    ### train the model
    model=obtain_model()
    model.summary()
    X_train,X_test,y_train,y_test=train_test_split(faces,y,test_size=0.2,
                                                   random_state=42,
                                                   shuffle = True)
    history = model.fit(x=X_train,y=y_train,epochs=50,validation_data=(X_test,y_test))
    ### plot accuracy and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    ### plot loss and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    ### predictions on whole dataset
    prediction = model.predict(faces)
    prediction = np.around(prediction)
    ### create confusion matrix as requested
    matrix     = confusion_matrix(y, prediction)
    print('confusion matrix')
    print(matrix)
    ### save the model
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
