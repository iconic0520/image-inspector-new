# Image Tampering Detection using Error Level Analysis (ELA)

This project provides a Streamlit web application for detecting image tampering using Error Level Analysis (ELA). It has been refactored from an original project to remove weather-checking functionality and ensure a streamlined ELA-focused experience.

## Features

*   **Error Level Analysis (ELA):** Detects inconsistencies in image compression, which can indicate tampering.
*   **Streamlit Interface:** A user-friendly web interface for uploading images and viewing ELA results.
*   **Model-based Detection:** Utilizes a pre-trained Keras model (`model_ela.h5`) for classifying images as \'Real\' or \'Tampered\' based on ELA features.

## Fixing the Model Error (Windows/Newer Python)

If you see an error like `Argument name must be a string and cannot contain character /`, follow these steps:

1.  Open your terminal in VS Code.
2.  Ensure your virtual environment is active.
3.  Run the fix script:
    ```bash
    python fix_model.py
    ```
4.  This will create a `model_ela_fixed.h5` file that works with modern TensorFlow. The app will automatically use this file.

## Training the Model

If you want to train your own model or improve the current one:

1.  **Get the Dataset:** Download the **CASIA 2.0** dataset. You can find it on [Kaggle](https://www.kaggle.com/datasets/sophiasophia/casia-20-image-tampering-detection-dataset).
2.  **Organize Files:** Place the images in the following structure:
    ```
    Casia2/
    ├── Real/      (Put authentic images here)
    └── Tampered/  (Put edited images here)
    ```
3.  **Run Training:**
    ```bash
    python ELA_Training/MainELA.py
    ```
    This will generate a new `model_ela_new.h5` file.

## Setup and Run Instructions (VS Code)

Follow these steps to set up and run the project in Visual Studio Code:

### Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.8+:** Download and install from [python.org](https://www.python.org/downloads/).
2.  **Visual Studio Code:** Download and install from [code.visualstudio.com](https://code.visualstudio.com/).
3.  **Git (Optional but Recommended):** For version control, download from [git-scm.com](https://git-scm.com/downloads).

### Step-by-Step Guide

1.  **Extract the Project:**
    *   Unzip the provided `image_tampering_detection_ela.zip` file to a directory of your choice (e.g., `C:\Projects\image_tampering_detection_ela` or `~/Projects/image_tampering_detection_ela`).

2.  **Open Project in VS Code:**
    *   Open VS Code.
    *   Go to `File` > `Open Folder...` (or `Code` > `Open Folder...` on macOS).
    *   Navigate to the extracted project folder (`image_tampering_detection_ela`) and click `Open`.

3.  **Create a Virtual Environment:**
    *   Open the integrated terminal in VS Code (`Terminal` > `New Terminal`).
    *   Create a virtual environment (recommended to isolate project dependencies):
        ```bash
        python3 -m venv venv
        ```

4.  **Activate the Virtual Environment:**
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    *   You should see `(venv)` at the beginning of your terminal prompt, indicating the virtual environment is active.

5.  **Install Dependencies:**
    *   With the virtual environment active, install the required Python packages:
        ```bash
        pip install -r requirements.txt
        ```
    *   This will install `streamlit`, `tensorflow`, `keras`, `opencv-python-headless`, `numpy`, `pillow`, `scikit-learn`, and `matplotlib`.

6.  **Run the Streamlit Application:**
    *   In the same terminal with the activated virtual environment, run the Streamlit app:
        ```bash
        streamlit run app.py
        ```

7.  **Access the Application:**
    *   After running the command, Streamlit will open a new tab in your default web browser, displaying the application. If it doesn\'t open automatically, it will provide a local URL (e.g., `http://localhost:8501`) that you can copy and paste into your browser.

8.  **Interact with the App:**
    *   Upload a `.jpg` or `.jpeg` image using the file uploader.
    *   Click the \"Proceed\" button to analyze the image for tampering using ELA.
    *   View the ELA analysis results and download the ELA image if desired.

## Project Structure

```
image_tampering_detection_ela/
├── app.py                  # Main Streamlit application
├── helper.py               # Helper functions for ELA processing
├── requirements.txt        # Python dependencies
├── ELA_Training/           # Contains the pre-trained ELA model
│   └── model_ela.h5        # Pre-trained Keras model for ELA classification
├── imgs/                   # Example images
├── rsc/                    # Resources like example ELA images
└── README.md               # This instruction file
```

## Notes

*   The `ELA_Training/MainELA.py` script is included for reference, showing how the ELA model was originally trained. It requires the CASIA2 dataset to run.
*   The weather detection functionality has been completely removed from this version of the project.
