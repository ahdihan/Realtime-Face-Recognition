import cv2
import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
import time

# Initialize face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def load_model():
    if os.path.exists("face_model.pkl"):
        with open("face_model.pkl", "rb") as f:
            knn_model, face_data, labels = pickle.load(f)
        return knn_model, face_data, labels
    else:
        return None, [], []


def save_model(knn_model, face_data, labels):
    with open("face_model.pkl", "wb") as f:
        pickle.dump((knn_model, face_data, labels), f)


def capture_faces(name, face_data, labels, num_samples=50):
    video = cv2.VideoCapture(0)
    captured_faces = []

    while len(captured_faces) < num_samples:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (100, 100))
            captured_faces.append(face_resized)

            cv2.putText(frame, f"Samples: {len(captured_faces)}/{num_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Capturing Faces', frame)

        # Delay to reduce frame capture speed
        time.sleep(0.3)  # Adds a 0.3-second delay to slow down frame capture

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    # Add new data
    for face in captured_faces:
        face_data.append(face.flatten())  # Flatten the image to a 1D vector
        labels.append(name)

    return face_data, labels


def train_model(face_data, labels):
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(face_data, labels)
    return knn_model


def find_matching_face(knn_model, face_resized):
    if knn_model is not None:
        pred = knn_model.predict([face_resized])
        confidence = knn_model.predict_proba([face_resized])
        return pred[0], max(confidence[0])
    return None, 0


def main():
    # Load existing model and data
    knn_model, face_data, labels = load_model()

    # Prompt for a new user
    while True:
        name = input("Enter the user's name: ")
        if name in labels:
            print(f"Username '{name}' is already taken. Please choose a different name.")
        else:
            break

    video = cv2.VideoCapture(0)
    print("Please face the camera to check for an existing face.")

    for _ in range(10):  # Give the user some time to face the camera
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (100, 100)).flatten()

                # Check if this face already exists in the model
                pred_name, confidence = find_matching_face(knn_model, face_resized)

                if confidence > 0.6:  # Confidence threshold to check face match
                    print(f"Existing user detected: {pred_name}")
                    video.release()
                    cv2.destroyAllWindows()
                    return

    print(f"New user: {name}")
    video.release()
    cv2.destroyAllWindows()

    # Capture and add new face data for the user
    face_data, labels = capture_faces(name, face_data, labels)

    # Retrain the model with updated data
    knn_model = train_model(face_data, labels)
    save_model(knn_model, face_data, labels)

    print(f"Training completed for {name}")


if __name__ == "__main__":
    main()
