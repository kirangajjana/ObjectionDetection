import cv2
import streamlit as st
import numpy as np
# object detection
def detect_objects(image):
    # Load pre-trained object detection model (e.g., using Haar cascades)
    # Replace with the appropriate path to your cascade file
    cascade_path = "path_to_cascade.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform object detection
    objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return objects

def draw_objects(image, objects):
    # Draw rectangles around the detected objects
    for (x, y, w, h) in objects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

def main():
    st.title("Object Detection with OpenCV")

    # File upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Perform object detection
        objects = detect_objects(image)

        # Draw rectangles around the detected objects
        output_image = draw_objects(image.copy(), objects)

        # Display the original and processed images
        st.subheader("Original Image")
        st.image(image, channels="BGR")

        st.subheader("Detected Objects")
        st.image(output_image, channels="BGR")

if __name__ == "__main__":
    main()
