import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
# IMPORTANT: Replace this with the correct, accessible path in your environment.
# In cloud deployments, you must load this file via st.file_uploader or package it with the app.
logo_template_path = "images/logo.png"

try:
    # Load Havells logo template (grayscale)
    logo_template = cv2.imread(logo_template_path, 0)
    if logo_template is None:
        st.error(f"Logo template image could not be loaded from '{logo_template_path}'. Please check the path.")
        # Create a mock template if loading fails to allow the app structure to run
        logo_template = np.zeros((100, 100), dtype=np.uint8)
        is_template_loaded = False
    else:
        is_template_loaded = True
        st.sidebar.info(f"Logo template loaded successfully ({logo_template.shape[1]}x{logo_template.shape[0]})")

except Exception as e:
    st.error(f"Error loading logo template: {e}. Cannot run logo detection.")
    logo_template = np.zeros((100, 100), dtype=np.uint8) # Fallback to prevent crash
    is_template_loaded = False


# Color range for Havells T-shirt (red in HSV)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

MIN_MATCH_COUNT = 15 # Increased minimum matches for stricter validation

# Initialize feature detector (using SIFT for better results than ORB, if available)
try:
    # SIFT is often proprietary; if it fails, fallback to ORB
    detector = cv2.SIFT_create()
except AttributeError:
    detector = cv2.ORB_create(nfeatures=5000)

if is_template_loaded:
    kp_template, des_template = detector.detectAndCompute(logo_template, None)
else:
    kp_template, des_template = None, None # Skip feature extraction if template failed


def detect_human(image):
    """Detects presence of a human face using Haar cascades."""
    # The image is usually rotated (check orientation in original image). We assume standard orientation for face detection.
    # Note: Haar cascades path requires the cv2 installation to have the data files.
    
    # Attempt to detect face
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # If no face is found, try rotating 90 degrees and checking again, since the input image may be sideways
    if len(faces) == 0:
        # Rotate image 90 degrees clockwise (or counter-clockwise, doesn't matter for face)
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        gray_rotated = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_rotated, 1.3, 5)
    
    # Returns True if a face is detected in either the original or rotated image
    return len(faces) > 0

def detect_tshirt_and_color(image):
    """Checks if a significant portion of the image is red (the T-shirt color)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    red_pixels = np.sum(mask > 0)
    total_pixels = image.shape[0] * image.shape[1]
    red_ratio = red_pixels / total_pixels
    # Adjusted threshold for overall red content
    return red_ratio > 0.20

def detect_and_box_logo(image, template_kp, template_des, template_img):
    """
    Detects the logo using feature matching and Homography, and returns the image
    with the auto-adjusted bounding box drawn.
    """
    if not is_template_loaded or template_des is None:
        st.warning("Logo detection skipped: Template not loaded or features missing.")
        return image, False

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect features in the current image (no fixed cropping)
    kp_image, des_image = detector.detectAndCompute(img_gray, None)

    if des_image is None or len(kp_image) < MIN_MATCH_COUNT:
        return image, False

    # Brute-Force Matcher setup
    if detector.__class__.__name__ == 'SIFT':
        # Flann-based matcher is better for SIFT/SURF
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(template_des, des_image, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    else:
        # BFMatcher for ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(template_des, des_image)
        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x: x.distance)
        # Take a reasonable number of the best matches
        good_matches = [m for m in matches if m.distance < 70] # Adjusted distance threshold for stricter matching

    st.sidebar.text(f"Features found in image: {len(kp_image)}")
    st.sidebar.text(f"Good matches for logo: {len(good_matches)}")

    if len(good_matches) > MIN_MATCH_COUNT:
        # Extract matched keypoints
        src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find Homography (H) - the transformation matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            h, w = template_img.shape
            # Define the corners of the template image
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            # Transform the template corners using Homography M to get the logo position in the image
            dst = cv2.perspectiveTransform(pts, M)

            # Draw the bounding box (using the transformed corners)
            image = cv2.polylines(image, [np.int32(dst)], True, (0, 255, 0), 4, cv2.LINE_AA)
            return image, True
        else:
            return image, False
    else:
        return image, False

# --- STREAMLIT APP LAYOUT ---
st.title("Havells T-shirt Logo Detection (Robust Feature Matching)")
st.markdown("Upload an image to check for Human presence, Red T-shirt color, and the Havells logo using auto-adjusting bounding box detection.")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    # Convert PIL image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.subheader("Processing Results")
    
    with st.spinner("Processing image (detecting features and homography)..."):
        human_detected = detect_human(img_cv)
        tshirt_correct = detect_tshirt_and_color(img_cv)
        
        # Core detection logic - draws the box and returns success state
        img_cv_boxed, logo_detected = detect_and_box_logo(
            img_cv, 
            kp_template, 
            des_template, 
            logo_template
        )

    # Display the image with the calculated bounding box
    img_vis = cv2.cvtColor(img_cv_boxed, cv2.COLOR_BGR2RGB)
    st.image(img_vis, caption="Uploaded Image with Auto-Adjusted Logo Highlight", use_column_width=True)

    st.markdown("---")
    st.metric("Human Detected", "Yes" if human_detected else "No")
    st.metric("T-shirt Color (Red)", "Correct" if tshirt_correct else "Incorrect")
    st.metric("Logo Detected & Bounding Box Drawn", "YES" if logo_detected else "NO")

    if human_detected and tshirt_correct and logo_detected:
        st.success("✅ Image meets all conditions!")
        if st.button("Save Image as Valid!"):
            # Note: This just saves the original file to a new name in the same execution context.
            with open("validated_image.png", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Original image saved as 'validated_image.png'!")
    else:
        st.warning("❌ Image does not meet all conditions.")
