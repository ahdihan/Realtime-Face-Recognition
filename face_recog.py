import cv2
import numpy as np
import pickle
import time


# Load the saved model
def load_model():
    with open("face_model.pkl", "rb") as f:
        knn_model, _, _ = pickle.load(f)
    return knn_model


# Initialize face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def main():
    knn_model = load_model()
    video = cv2.VideoCapture(0)
    max_confidence = 0
    predicted_name = "Unrecognized"

    start_time = time.time()  # Get the starting time

    while time.time() - start_time < 5:  # Run the face detection for 5 seconds
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (100, 100)).flatten()

            # Predict the face
            pred = knn_model.predict([face_resized])
            confidence = knn_model.predict_proba([face_resized])
            confidence_value = max(confidence[0])

            if confidence_value > max_confidence:
                max_confidence = confidence_value
                predicted_name = pred[0]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    if max_confidence > 0.6:  # Confidence threshold
        print(f"User recognized: {predicted_name}")
    else:
        print("Unrecognized face")


if __name__ == "__main__":
    main()
