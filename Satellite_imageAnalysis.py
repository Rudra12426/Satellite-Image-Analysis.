"""
Satellite Image Analysis using Color Segmentation
Dataset: EuroSAT
Author: Rudra Pratap
"""

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# ----------------------------
# Dataset Path (Relative Path for GitHub)
# ----------------------------
dataset_path = "dataset"

# ----------------------------
# Get All Image Files
# ----------------------------
def get_image_files(path):
    return [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# ----------------------------
# Preprocess Image
# ----------------------------
def preprocess_image(img, size=(256, 256)):
    img_resized = cv2.resize(img, size)
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    return img_resized, img_hsv

# ----------------------------
# Color Segmentation
# ----------------------------
def segment_colors(img_hsv):
    lower_blue = np.array([85, 50, 50])
    upper_blue = np.array([135, 255, 255])
    mask_water = cv2.inRange(img_hsv, lower_blue, upper_blue)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask_veg = cv2.inRange(img_hsv, lower_green, upper_green)

    lower_urban = np.array([0, 0, 50])
    upper_urban = np.array([180, 60, 200])
    mask_urban = cv2.inRange(img_hsv, lower_urban, upper_urban)

    return mask_water, mask_veg, mask_urban

# ----------------------------
# Display Results
# ----------------------------
def display_results(img_resized, masks, img_name):
    mask_water, mask_veg, mask_urban = masks

    plt.figure(figsize=(12,4))

    plt.subplot(1,4,1)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title(f"{img_name}")
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.imshow(mask_water, cmap='Blues')
    plt.title("Water Mask")
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.imshow(mask_veg, cmap='Greens')
    plt.title("Vegetation Mask")
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.imshow(mask_urban, cmap='gray')
    plt.title("Urban Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    if not os.path.exists(dataset_path):
        print("Dataset folder not found!")
        exit()

    image_files = get_image_files(dataset_path)
    print(f"Found {len(image_files)} images.")

    num_to_process = 10

    for file_name in image_files[:num_to_process]:
        img_path = os.path.join(dataset_path, file_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to load: {file_name}")
            continue

        img_resized, img_hsv = preprocess_image(img)
        masks = segment_colors(img_hsv)
        display_results(img_resized, masks, file_name)    

# Advanced Version Of this Project
# Satellite Image Analysis – EuroSAT (Automated Full Analysis)
import cv2
import os
import numpy as np
import pandas as pd

# ----------------------------
# 1️⃣ Dataset Path
# ----------------------------
dataset_path = "Dataset"
output_path = os.path.join(dataset_path, "Output")
os.makedirs(output_path, exist_ok=True)

# Create folders for masks
mask_folders = ["Water", "Vegetation", "Urban"]
for folder in mask_folders:
    os.makedirs(os.path.join(output_path, folder), exist_ok=True)

# ----------------------------
# 2️⃣ Get All Image Files
# ----------------------------
def get_image_files(dataset_path):
    return [f for f in os.listdir(dataset_path) if f.lower().endswith(".jpg")]

# ----------------------------
# 3️⃣ Preprocess Image
# ----------------------------
def preprocess_image(img, size=(256, 256)):
    img_resized = cv2.resize(img, size)
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    return img_resized, img_hsv

# ----------------------------
# 4️⃣ Color Segmentation
# ----------------------------
def segment_colors(img_hsv):
    # Water mask (blue)
    lower_blue = np.array([85, 50, 50])
    upper_blue = np.array([135, 255, 255])
    mask_water = cv2.inRange(img_hsv, lower_blue, upper_blue)

    # Vegetation mask (green)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask_veg = cv2.inRange(img_hsv, lower_green, upper_green)

    # Urban mask (gray/red/brown)
    lower_urban = np.array([0, 0, 50])
    upper_urban = np.array([180, 60, 200])
    mask_urban = cv2.inRange(img_hsv, lower_urban, upper_urban)

    return mask_water, mask_veg, mask_urban

# ----------------------------
# 5️⃣ Save Masks
# ----------------------------
def save_masks(img_name, masks):
    mask_water, mask_veg, mask_urban = masks
    cv2.imwrite(os.path.join(output_path, "Water", img_name), mask_water)
    cv2.imwrite(os.path.join(output_path, "Vegetation", img_name), mask_veg)
    cv2.imwrite(os.path.join(output_path, "Urban", img_name), mask_urban)

# ----------------------------
# 6️⃣ Calculate % Coverage
# ----------------------------
def calculate_coverage(mask):
    total_pixels = mask.size
    white_pixels = cv2.countNonZero(mask)
    percent = (white_pixels / total_pixels) * 100
    return round(percent, 2)

# ----------------------------
# 7️⃣ Run Analysis
# ----------------------------
if __name__ == "__main__":
    image_files = get_image_files(dataset_path)
    if not image_files:
        print("No images found in folder!")
        exit()

    print(f"Processing {len(image_files)} images...")

    # Create a list to store results for CSV
    results = []

    for idx, file_name in enumerate(image_files):
        img_path = os.path.join(dataset_path, file_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load: {file_name}")
            continue

        img_resized, img_hsv = preprocess_image(img)
        masks = segment_colors(img_hsv)

        # Save masks
        save_masks(file_name, masks)

        # Calculate % coverage
        water_pct = calculate_coverage(masks[0])
        veg_pct = calculate_coverage(masks[1])
        urban_pct = calculate_coverage(masks[2])

        # Append results
        results.append({
            "Image": file_name,
            "Water_%": water_pct,
            "Vegetation_%": veg_pct,
            "Urban_%": urban_pct
        })

        if (idx+1) % 50 == 0:
            print(f"Processed {idx+1} images...")

    # Export CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_path, "EuroSAT_Analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"Analysis complete! CSV saved at: {csv_path}")
