import streamlit as st
import cv2
import os
import numpy as np
from keras.models import load_model
from helper import prepare_image_for_ela
from srm_analysis import apply_srm, get_srm_score
from PIL import Image as PILImage
import io

# Constants
class_ELA = ['Real', 'Tampered']

# Functions
def check_img(image_name):
    img = cv2.resize(cv2.imread(image_name), (750, 750))
    return img

def detect_ELA(img_name):
    global ela_result
    np_img_input, ela_result = prepare_image_for_ela(img_name)
    # Load model once or within function. For simplicity, we keep it here.
    # In a production app, loading once outside is better.
    # Use the fixed model if it exists, otherwise use the original
    model_path = 'ELA_Training/model_ela_fixed.h5'
    if not os.path.exists(model_path):
        model_path = 'ELA_Training/model_ela.h5'
    
    # Ensure paths are correct for the current OS
    model_path = os.path.join(*model_path.split('/'))
    
    try:
        ELA_model = load_model(model_path)
    except Exception as e:
        # Fallback for newer Keras: sometimes we need to rebuild the architecture
        # or load the model with custom settings.
        st.error(f"Failed to load model from {model_path}. Error: {e}")
        st.info("Try running 'python fix_model.py' first to fix model compatibility.")
        raise e
    Y_predicted = ELA_model.predict(np_img_input, verbose=0)
    accuracy = round(np.max(Y_predicted[0]) * 100)
    prediction = class_ELA[np.argmax(Y_predicted[0])]
    return f"Model shows {accuracy}% confidence of image being {prediction}"

# Streamlit Interface
st.set_page_config(page_title="Image Tampering Detection", layout="wide")
st.title("Image Tampering Detection Using ELA and SRM [Noise inconsistencies]")

st.markdown(""" 
            ### Welcome to the Image Tampering Detector!
            **What This Project Does:**
            This tool helps you determine whether an image has been altered or manipulated using **Error Level Analysis (ELA)** and **Steganalysis Rich Model (SRM)**.
            """)

with st.expander("How it Works?"):
    st.markdown("""
    **Error Level Analysis (ELA):**
    Images that are edited often show different compression artifacts compared to the original ones. 
    ELA highlights these discrepancies, making tampering visible.
        
    **Steganalysis Rich Model (SRM):**
    It is a feature extraction technique used in image tampering detection that captures subtle statistical noise patterns in an image to identify hidden modifications or manipulations.
    """)

    # Create columns for horizontal layout
    col1, col2 = st.columns(2)

    # Display ELA images in columns
    with col1:
        if os.path.exists("rsc/real.jpg"):
            st.image("rsc/real.jpg", caption="ELA of a Real Image", use_container_width=True)
        else:
            st.info("Example 'real' image not found in rsc/")

    with col2:
        if os.path.exists("rsc/fake.jpg"):
            st.image("rsc/fake.jpg", caption="ELA of a Tampered Image", use_container_width=True)
        else:
            st.info("Example 'fake' image not found in rsc/")

    st.markdown("""
    Notice how ELA on a real image shows consistency across the image, but on a tampered image, 
    ELA shows discrepancies (brighter or different patterns) across some regions.
    """)

with st.expander("How to use it?"):
    st.markdown("""
    1. **Upload an Image:** Choose a .jpg or .jpeg image to analyze.
    2. **View Results:** Get insights into the image’s authenticity with ELA analysis.
    3. **Save ELA:** You can download the ELA result for further inspection.
    4. **Save SRM:** You can download the SRM result for further inspection.
    """)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0

# Handle file upload
uploaded_file = st.file_uploader("Choose a .jpg/.jpeg image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    temp_path = os.path.join(os.getcwd(), "temp_upload.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Proceed"):
        st.session_state.step = 1

# Show results if step > 0
if st.session_state.step > 0 and uploaded_file is not None:
    st.write("### Analysis Results:")
    
    with st.spinner('Analyzing...'):
        try:
            # Use absolute path for Windows compatibility
            temp_path = os.path.join(os.getcwd(), "temp_upload.jpg")
            res1 = detect_ELA(temp_path)
            st.success(res1)
            
            # Display ELA result image
            st.image(ela_result, caption="ELA Analysis Result", use_container_width=True)
            
            # --- Added SRM Analysis ---
            st.write("### Forensic Layer 2: Noise Residual Analysis (SRM)")
            with st.spinner('Analyzing noise residuals...'):
                srm_img = apply_srm(temp_path)
                srm_score = get_srm_score(temp_path)
                
                # Display SRM result
                st.image(srm_img, caption="SRM Analysis Result (Noise Residuals)", use_container_width=True)
                
                # Heuristic logic for SRM score
                if srm_score > 1000: # Example threshold
                    st.warning(f"High Noise Variance detected: {srm_score:.2f}. This may indicate localized editing or noise injection.")
                else:
                    st.info(f"Normal Noise Variance detected: {srm_score:.2f}. Noise distribution appears consistent.")
            
            # Save ELA result image to a BytesIO object for download
            buffer = io.BytesIO()
            ela_result.save(buffer, format="JPEG")
            buffer.seek(0)
            
            st.download_button(
                label="Download ELA Result Image",
                data=buffer,
                file_name="ela_result.jpg",
                mime="image/jpeg"
            )
        except Exception as e:
            st.error(f"Error during analysis: {e}")

    # Button to reset
    if st.button("Try New Image"):
        st.session_state.step = 0
        st.rerun()
