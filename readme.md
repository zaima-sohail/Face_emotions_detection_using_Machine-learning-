# Face emotion recognition

A Streamlit app for running a trained Keras image-classification model on a
webcam photo, one uploaded image, or a batch of uploaded images. The app detects
a face, prepares a grayscale 48 × 48 crop, and displays the model's predicted
class and output probabilities.
<img width="1908" height="993" alt="image" src="https://github.com/user-attachments/assets/01c62322-55ec-418a-baa6-804d915c5661" />


# live url:https://zaima-sohail-face-emotions-detection-us-face-recognition-ndxaxb.streamlit.app/
> **Important:** This is an experimental facial-expression classifier. It
> cannot reliably determine how a person feels. Predictions and confidence
> values are model outputs, not calibrated measures of emotional state.

## Features

- **Webcam:** capture a still photo in the browser and analyze it. This is not
  continuous live-video detection.
- **Single image:** upload one JPG, JPEG, PNG, BMP, or WebP image.
- **Batch upload:** upload multiple supported images, compare per-image
  predictions, inspect class-count and average-probability charts, and download
  the results as CSV.
- **Face detection:** locate faces with OpenCV's frontal-face cascade. If more
  than one face is found, the largest face is used.
- **Model selection:** choose a Keras `.h5` model using the sidebar's **Model
  file** field.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or later
- A webcam for webcam capture
- The project model file `emotional_detector.h5`
- The training label folders described under [Model and class labels](#model-and-class-labels)

## Install

Open PowerShell in the project directory:

```powershell
cd C:\Users\zaima\face_recognition
```

Install the Python dependencies:

```powershell
& "C:\Users\zaima\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -r requirements.txt
```

If Python 3.10 is installed in a different location, replace that executable
path with the Python interpreter you use for this project. The requirements
include Streamlit, TensorFlow/Keras, OpenCV, Pillow, NumPy, and pandas.

## Run the app

From the project directory, run:

```powershell
& "C:\Users\zaima\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run face_recognition.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in your
browser. In **Webcam** mode, allow the browser to access your camera and take a
photo. Camera access is controlled by the browser and generally requires
`localhost` or a secure HTTPS connection.

## Use

1. Start the app using the command above.
2. If the model is not at the default path, enter its path in the sidebar's
   **Model file** field.
3. Choose **Webcam**, **Single image**, or **Batch upload**.
4. Provide a clear image with a visible, front-facing face.
5. Review the prediction and probability chart. In batch mode, download the
   full result table with **Download predictions as CSV**.

Images with no detectable face or unreadable image data are reported rather
than sent to the classifier. In batch mode, other valid images continue to be
processed.

## Project files

```text
face_recognition/
├── face_recognition.py       # Streamlit application
├── emotional_detector.h5     # Trained Keras model used by default
├── model.json                # Serialized model architecture
├── face_detection.ipynb      # Training and experimentation notebook
├── requirements.txt          # Python dependencies
└── archive (3)/
    └── images/images/train/  # Label folders used to display class names
```

The app reads label names from the directories inside
`archive (3)/images/images/train/`, in alphabetical order.

## Model and class labels

The included model expects grayscale images with shape `(48, 48, 1)`. The app
detects and crops the largest face, resizes it to 48 × 48, and scales pixel
values to the range 0–1 before inference.

**The included artifacts currently disagree about the number of classes:**
`emotional_detector.h5` has 7 output units, while the available training
folders are `angry`, `disgust`, `fear`, and `happy` (4 classes). The app names
the first four outputs from those folders and marks the remaining three
outputs as **unmapped**. An unmapped output must not be interpreted as a known
emotion.

For meaningful class labels, retrain the model with the same number of output
units as training classes and save a model whose output order exactly matches
the label encoder's class order. Then point the app to that model. Do not
rename the extra outputs to `sad`, `surprise`, or `neutral` unless those
classes were actually present in the training data and used to train the
corresponding output units.

## Troubleshooting

### Model file not found

Confirm that `emotional_detector.h5` is in the project folder, or set the full
path to the model in the sidebar.

### Could not initialize face detection

Install dependencies from `requirements.txt` using the same Python interpreter
used to launch Streamlit. The project pins OpenCV to a version range that
includes its bundled face-detection cascade.

### No face detected

Try a well-lit, front-facing photo with the face unobstructed and closer to the
camera. The app currently uses a frontal-face detector and may not recognize
side profiles or very small faces.

### Camera widget is unavailable

Allow camera permissions for the page in the browser. Use `localhost` for
local development or serve the app over HTTPS for remote access.

### Import or package errors

Install `requirements.txt` with the same Python interpreter used to run
Streamlit. For example:

```powershell
& "C:\Users\zaima\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -r requirements.txt
```

## Privacy and responsible use

Webcam and uploaded image contents are processed by the running Python app to
produce predictions; the app does not intentionally save images to disk.
Handle images in accordance with the consent and privacy expectations of the
people depicted. Do not use this experimental classifier for consequential
decisions, diagnosis, surveillance, or claims about a person's internal
emotional state.
