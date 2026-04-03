# Import required libraries for emotion prediction from a single image
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained emotion detection model
model = load_model("models/emotion_model.h5")

# Define emotion labels in the same order as training classes
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Give the path of the test image
image_path = "test_images/test1.jpeg"

# Read the image
img = cv2.imread(image_path)

# Check if image loaded properly
if img is None:
    print("Error: Image not found. Check the image path.")
    exit()

# Convert image to grayscale because model expects grayscale face input
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Check if any face is detected
if len(faces) == 0:
    print("No face detected in the image.")
    cv2.imshow("Emotion Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# Process each detected face
for (x, y, w, h) in faces:
    # Crop the face region from grayscale image
    face = gray[y:y+h, x:x+w]

    # Resize face to 48x48 because model was trained on 48x48 images
    face = cv2.resize(face, (48, 48))

    # Normalize pixel values from 0-255 to 0-1
    face = face / 255.0

    # Reshape image to match model input shape: (1, 48, 48, 1)
    face = np.reshape(face, (1, 48, 48, 1))

    # Predict emotion probabilities
    prediction = model.predict(face, verbose=0)

    # Get index of highest probability
    predicted_index = np.argmax(prediction)

    # Get emotion label
    predicted_emotion = emotion_labels[predicted_index]

    # Get confidence score
    confidence = np.max(prediction) * 100

    # Draw rectangle around detected face
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Put predicted emotion text above face
    text = f"{predicted_emotion} ({confidence:.2f}%)"
    cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Print result in terminal also
    print("Predicted Emotion:", predicted_emotion)
    print("Confidence:", f"{confidence:.2f}%")

# Show output image
cv2.imshow("Emotion Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()