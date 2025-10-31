import cv2
import numpy as np
import os

# Paths of your sample images
image_paths = [
    "E:/havells/IMG_20251030_162056.jpg"
]

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
    # Crop likely pocket area where logo appears, tune coordinates as per data
    pocket_crop = image[int(0.5*h):int(0.78*h), int(0.55*w):int(0.93*w)]
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
    return len(good_matches) > 30  # Tune threshold as needed

def process_images(image_paths, output_folder="filtered_results"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load {path}")
            continue
        if not detect_human(img):
            print(f"No human: {path}")
            continue
        if not detect_tshirt_and_color(img):
            print(f"Not matching T-shirt color: {path}")
            continue
        if not detect_logo_orb(img):
            print(f"Logo not detected: {path}")
            continue
        out_path = os.path.join(output_folder, os.path.basename(path))
        cv2.imwrite(out_path, img)
        print(f"Image saved: {out_path}")

# Run the workflow
process_images(image_paths)
