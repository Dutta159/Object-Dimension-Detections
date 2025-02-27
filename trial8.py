import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import os

def save_dimensions_to_excel(width_m, height_m, area_m, perimeter_m):
    file_name = "object_dimensions.xlsx"
    columns = ["Width (m)", "Height (m)", "Area (m^2)", "Perimeter (m)"]
    data = pd.DataFrame({
        "Width (m)": [width_m],
        "Height (m)": [height_m],
        "Area (m^2)": [area_m],
        "Perimeter (m)": [perimeter_m]
    })
    
    if os.path.exists(file_name):
        existing_data = pd.read_excel(file_name)
        data = pd.concat([existing_data, data], ignore_index=True)
    
    data.to_excel(file_name, index=False)

def calculate_pixels_per_meter(reference_image, real_width_m):
    gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        width = max(rect[1])
        pixels_per_meter = width / real_width_m
        return pixels_per_meter
    return None

def calculate_object_dimensions_from_frame(frame, pixels_per_meter):
    original = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        cv2.drawContours(original, [box], 0, (0, 255, 0), 2)
        
        width_m = max(rect[1]) / pixels_per_meter
        height_m = min(rect[1]) / pixels_per_meter
        object_area = (rect[1][0] * rect[1][1]) / (pixels_per_meter ** 2)
        object_perimeter = (2 * (rect[1][0] + rect[1][1])) / pixels_per_meter
        
        save_dimensions_to_excel(width_m, height_m, object_area, object_perimeter)
        
        cv2.putText(original, f"Width: {width_m:.3f}m", (int(rect[0][0]), int(rect[0][1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(original, f"Height: {height_m:.3f}m", (int(rect[0][0]), int(rect[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(original, f"Area: {object_area:.3f}m^2", (int(rect[0][0]), int(rect[0][1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(original, f"Perimeter: {object_perimeter:.3f}m", (int(rect[0][0]), int(rect[0][1]) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return original

def main():
    st.title("Live Object Measurement with Streamlit")
    
    st.subheader("Upload a Reference Image")
    ref_image = st.file_uploader("Upload an image with a known object size", type=["jpg", "png", "jpeg"])
    real_width_m = st.number_input("Enter the real-world width of the object in meters", min_value=0.01, step=0.01)
    
    if ref_image and real_width_m:
        image = Image.open(ref_image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        pixels_per_meter = calculate_pixels_per_meter(image, real_width_m)
        if pixels_per_meter:
            st.success(f"Calibration complete: {pixels_per_meter:.2f} pixels per meter")
            st.session_state.pixels_per_meter = pixels_per_meter
        else:
            st.error("Could not detect object in reference image.")
            return
    
    if "pixels_per_meter" not in st.session_state:
        st.warning("Please upload a reference image and enter the real-world size to start.")
        return
    
    if "run" not in st.session_state:
        st.session_state.run = False
    
    if st.button("Start Stream"):
        st.session_state.run = True
    if st.button("Stop Stream"):
        st.session_state.run = False
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")
        return
    
    stframe = st.empty()
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break
        
        processed_frame = calculate_object_dimensions_from_frame(frame, st.session_state.pixels_per_meter)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(processed_frame)
        stframe.image(img, caption="Processed Frame", use_container_width=True)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
