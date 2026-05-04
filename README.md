# Image Tampering Detection using ELA and SRM Analysis

This project provides a professional Streamlit web application for detecting image tampering using a dual-layered forensic approach: **Error Level Analysis (ELA)** and **Steganalysis Rich Model (SRM) Noise Analysis**.

## 🚀 Features

*   **Error Level Analysis (ELA):** Detects inconsistencies in image compression levels, highlighting areas that have been modified and re-saved.
*   **Noise Residual Analysis (SRM):** Extracts high-frequency noise residuals to identify disruptions in the camera sensor's natural "fingerprint," making it highly effective at spotting spliced objects.
*   **AI-Powered Verdict:** Utilizes a pre-trained CNN model (trained on the CASIA 2.0 dataset) to provide an automated "Real" or "Tampered" classification.
*   **Metadata Auditing:** Scans EXIF data for traces of editing software like Adobe Photoshop or GIMP.
*   **Interactive Dashboard:** A clean Streamlit interface for uploading images, viewing forensic maps, and downloading results.

## 🔍 Advanced Forensic Layer: SRM Analysis

Unlike ELA, which focuses on compression, **SRM (Steganalysis Rich Model)** focuses on **Noise Inconsistency**. 
*   **How it helps:** When an image is edited, the natural noise pattern is broken. SRM highlights these breaks, providing a second layer of proof that is much harder for forgers to hide.
*   **Dual-Verification:** If both ELA and SRM highlight the same area, the probability of tampering is near 100%.

## 🛠️ Setup and Run Instructions (VS Code)

### Prerequisites
1.  **Python 3.8+:** Download from [python.org](https://www.python.org/ ).
2.  **VS Code:** Download from [code.visualstudio.com](https://code.visualstudio.com/ ).

### Step-by-Step Guide

1.  **Open Project in VS Code:**
    Navigate to your project folder and open it in VS Code.

2.  **Create and Activate Virtual Environment:**
    Open the terminal (`Ctrl + \``) and run:
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Install the required packages, including the new forensic libraries:
    ```bash
    pip install -r requirements.txt
    pip install opencv-python-headless
    ```

4.  **Fix the Model (Required for Windows/Newer Python):**
    If you encounter a Keras loading error, run the fix script:
    ```bash
    python fix_model.py
    ```

5.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure
    image-inspector-new/
├── ELA_Training/           # Pre-trained Keras model & training scripts

├── imgs/                   # Sample images for testing

├── res/                    # Results and output storage

├── rsc/                    # Resource files and assets

├── app.py                  # Main Streamlit web application

├── srm_analysis.py         # SRM Noise Residual Analysis module

├── helper.py               # ELA processing and metadata helper functions

├── fix_model.py            # Script to fix model compatibility for Windows/Keras

├── requirements.txt        # List of required Python packages

├── README.md               # Project documentation and instructions

├── analyze_user_img.py     # Script for direct command-line image analysis

├── test_ela.py             # Unit test script for ELA functionality

└── temp_upload.jpg         # Temporary file for processed uploads




