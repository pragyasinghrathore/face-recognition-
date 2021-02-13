import cv2 as cv
import numpy as np
from keras.models import model_from_yaml

def load_model():
    # load YAML and create model
    yaml_file = open('model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model
    
def use_webcam(model):
    ### webcam usage
    # use classifier
    face_cascade = cv.CascadeClassifier('faces_classifier.xml')
    cap = cv.VideoCapture(0)
    smile_probability = 0.0
    if not cap.isOpened():
        exit()
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face=face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(face)>0:
            # crop image
            (x,y,w,h)=face[0]
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), 2)
            face_frame=frame[y:(y+h),x:(x+w)]
            resized_face_frame=cv.resize(face_frame, (64, 64), interpolation=cv.INTER_CUBIC)
            # make prediction
            prediction=loaded_model.predict(np.asanyarray([resized_face_frame]))
            smile_probability = prediction[0][0]
        # put text
        if smile_probability>0.55:
            cv.putText(frame,"smile",(220,40),cv.FONT_HERSHEY_SIMPLEX,0.77,(0,255,0))
        else:
            cv.putText(frame,"no smile",(220, 80),cv.FONT_HERSHEY_SIMPLEX,0.77,(0,3,255))
        cv.imshow('frame', frame)
        # exit function
        if cv.waitKey(1) == ord('q'):
            break
    # close everything
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    ### load saved model
    loaded_model = load_model()
    ### make predictions
    use_webcam(loaded_model)