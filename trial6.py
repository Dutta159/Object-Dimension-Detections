import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import os

def save_dimensions_to_excel(width, height, area, perimeter):
    file_name = "object_dimensions.xlsx"
    columns = ["Width (px)", "Height (px)", "Area (px^2)", "Perimeter (px)"]
    data = pd.DataFrame({
        "Width (px)": [width],
        "Height (px)": [height],
        "Area (px^2)": [area],
        "Perimeter (px)": [perimeter]
    })
    
    if os.path.exists(file_name):
        existing_data = pd.read_excel(file_name)
        data = pd.concat([existing_data, data], ignore_index=True)
    else:
        data.to_excel(file_name, index=False, header=True)
        return
    
    data.to_excel(file_name, index=False)

def calculate_object_dimensions_from_frame(frame):
    original = frame.copy()

    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area to focus on the largest contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        # Get the convex hull of the largest contour
        hull = cv2.convexHull(contours[0])
        cv2.drawContours(original, [hull], -1, (0, 0, 255), 2)

        # Get the bounding box of the convex hull
        x, y, w, h = cv2.boundingRect(hull)
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Compute the area and perimeter of the convex hull
        object_area = cv2.contourArea(hull)
        object_perimeter = cv2.arcLength(hull, True)
        
        # Draw the minimum area rectangle
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        cv2.drawContours(original, [box], 0, (255, 0, 0), 2)
        
        # Save dimensions to Excel
        save_dimensions_to_excel(w, h, object_area, object_perimeter)
        
        # Display dimensions on the frame
        cv2.putText(original, f"Width: {w}px", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(original, f"Height: {h}px", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(original, f"Area: {int(object_area)}px^2", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(original, f"Perimeter: {int(object_perimeter)}px", (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return original

def main():
    st.title("Live Object Measurement with Streamlit")
    
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
        
        processed_frame = calculate_object_dimensions_from_frame(frame)
        
        # Convert to RGB for Streamlit
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(processed_frame)
        
        stframe.image(img, caption="Processed Frame", use_container_width=True)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
