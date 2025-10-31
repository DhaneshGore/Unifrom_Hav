import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os


# Load Havells logo template (grayscale)
logo_template_path = "E:/havells/Gemini_Generated_Image_iv8ph9iv8ph9iv8p1.png"
logo_template = cv2.imread(logo_template_path, 0)
if logo_template is None:
    raise FileNotFoundError(f"Logo template image could not be loaded from {logo_template_path}")
print("Loaded logo template shape:", logo_template.shape)

# Color range for Havells T-shirt (red in HSV)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

def detect_human(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

def detect_tshirt_and_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    red_pixels = np.sum(mask > 0)
    total_pixels = image.shape[0] * image.shape[1]
    red_ratio = red_pixels / total_pixels
    return red_ratio > 0.15  # Adjust threshold as needed

def detect_logo_orb(image):
    h, w = image.shape[:2]
    # Crop pocket region expected to contain logo
    x1, y1 = int(0.56 * w), int(0.72 * h)
    x2, y2 = int(0.83 * w), int(0.82 * h)
    pocket_crop = image[y1:y2, x1:x2]
    gray_img = cv2.cvtColor(pocket_crop, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(logo_template, None)
    kp2, des2 = orb.detectAndCompute(gray_img, None)
    if des1 is None or des2 is None:
        return False
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 60]
    print(f"Good matches count (pocket area): {len(good_matches)}")
    return len(good_matches) > 40

st.title("Havells T-shirt Logo Detection")

uploaded_file = st.file_uploader("Upload a photo or use your webcam", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Draw pocket crop rectangle on the image for visualization
    h, w = img_cv.shape[:2]
    x1, y1 = int(0.60 * w), int(0.55 * h)
    x2, y2 = int(0.90 * w), int(0.75 * h)
    img_cv_vis = img_cv.copy()
    cv2.rectangle(img_cv_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle

    img_vis = cv2.cvtColor(img_cv_vis, cv2.COLOR_BGR2RGB)
    st.image(img_vis, caption="Uploaded Image with Logo Region Highlight", use_column_width=True)

    with st.spinner("Processing image..."):
        human_detected = detect_human(img_cv)
        tshirt_correct = detect_tshirt_and_color(img_cv)
        logo_detected = detect_logo_orb(img_cv)

    st.write("Human detected:", human_detected)
    st.write("T-shirt color correct:", tshirt_correct)
    st.write("Logo detected:", logo_detected)

    if human_detected and tshirt_correct and logo_detected:
        if st.button("Save Image as Valid!"):
            with open("validated_image.png", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Image saved successfully!")
    else:
        st.warning("Image does not meet all conditions. Please try another photo.")
