from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "emotional_detector.h5"
TRAIN_DIR = APP_DIR / "archive (3)" / "images" / "images" / "train"
IMAGE_SIZE = (48, 48)
SUPPORTED_TYPES = ("jpg", "jpeg", "png", "bmp", "webp")


st.set_page_config(
    page_title="Emotion recognition",
    page_icon=":material/mood:",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading emotion model...")
def load_model(model_path: str):
    from keras.models import load_model as keras_load_model

    return keras_load_model(model_path, compile=False)


@st.cache_resource
def load_face_detector():
    detector = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    if detector.empty():
        raise RuntimeError("OpenCV's frontal-face detection model could not be loaded.")
    return detector


def get_class_names(output_count: int) -> list[str]:
    folder_names = sorted(
        folder.name
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    ) if TRAIN_DIR.is_dir() else []

    names = folder_names[:output_count]
    names.extend(
        f"unmapped output {index + 1}"
        for index in range(len(names), output_count)
    )
    return names


def preprocess_image(image_bytes: bytes) -> tuple[Image.Image, np.ndarray]:
    with Image.open(BytesIO(image_bytes)) as source:
        rgb_image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))

    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    faces = load_face_detector().detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24),
    )
    if len(faces) == 0:
        raise ValueError("No face detected. Use a clear, front-facing photo.")

    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    face_image = Image.fromarray(
        gray_image[y : y + height, x : x + width]
    ).resize(IMAGE_SIZE, Image.Resampling.NEAREST)
    preview_array = rgb_image.copy()
    cv2.rectangle(
        preview_array,
        (x, y),
        (x + width, y + height),
        color=(0, 200, 80),
        thickness=3,
    )
    array = np.asarray(face_image, dtype=np.float32) / 255.0
    preview = Image.fromarray(preview_array)
    return preview, array[np.newaxis, ..., np.newaxis]


def predict_images(model, uploads, class_names: list[str]):
    valid_uploads = []
    tensors = []
    errors = []

    for upload in uploads:
        try:
            _, tensor = preprocess_image(upload.getvalue())
        except (UnidentifiedImageError, OSError, ValueError) as error:
            errors.append({"File": upload.name, "Error": str(error)})
            continue
        valid_uploads.append(upload)
        tensors.append(tensor[0])

    if not tensors:
        return pd.DataFrame(), errors

    predictions = model.predict(np.stack(tensors), verbose=0)
    if predictions.ndim != 2 or predictions.shape[1] != len(class_names):
        raise ValueError(
            f"Model returned shape {predictions.shape}; expected "
            f"(number of images, {len(class_names)})."
        )

    rows = []
    for upload, probabilities in zip(valid_uploads, predictions):
        predicted_index = int(np.argmax(probabilities))
        row = {
            "File": upload.name,
            "Prediction": class_names[predicted_index],
            "Confidence": float(probabilities[predicted_index]),
        }
        row.update(
            {
                f"Probability: {name}": float(probability)
                for name, probability in zip(class_names, probabilities)
            }
        )
        rows.append(row)

    return pd.DataFrame(rows), errors


st.title("Emotion recognition")
st.write("Capture a webcam photo, classify one image, or compare a batch.")

with st.sidebar:
    st.header("Model settings")
    model_path_input = st.text_input(
        "Model file",
        value=str(DEFAULT_MODEL_PATH),
        help="Path to a trained Keras .h5 model.",
    )

model_path = Path(model_path_input).expanduser()
if not model_path.is_file():
    st.error(f"Model file not found: {model_path}")
    st.stop()

try:
    model = load_model(str(model_path.resolve()))
except (OSError, ValueError, ImportError) as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

try:
    load_face_detector()
except RuntimeError as error:
    st.error(f"Could not initialize face detection: {error}")
    st.stop()

output_shape = model.output_shape
if not isinstance(output_shape, tuple) or output_shape[-1] is None:
    st.error(f"Could not determine the model's output classes: {output_shape}")
    st.stop()

class_names = get_class_names(int(output_shape[-1]))
known_class_count = min(
    len(
        [
            folder
            for folder in TRAIN_DIR.iterdir()
            if folder.is_dir()
        ]
    ) if TRAIN_DIR.is_dir() else 0,
    len(class_names),
)

if known_class_count < len(class_names):
    st.warning(
        f"This model has {len(class_names)} output units, but only "
        f"{known_class_count} training class folders were found. The extra "
        "outputs are shown as unmapped, not as emotion labels. Retrain with "
        "the correct number of output units and save the matching class order."
    )

mode = st.segmented_control(
    "Prediction mode",
    options=["Webcam", "Single image", "Batch upload"],
    default="Single image",
)

if mode in ("Webcam", "Single image"):
    if mode == "Webcam":
        st.info(
            "Take a webcam photo to analyze it. This captures a still image; "
            "it does not continuously analyze live video."
        )
        upload = st.camera_input(
            "Take a clear, front-facing photo",
            key="webcam_photo",
            resolution="720p",
        )
        image_label = "Webcam capture"
    else:
        upload = st.file_uploader(
            "Choose an image",
            type=SUPPORTED_TYPES,
            key="single_image",
        )
        image_label = upload.name if upload is not None else ""

    if upload:
        try:
            preview, tensor = preprocess_image(upload.getvalue())
        except (UnidentifiedImageError, OSError, ValueError) as error:
            st.error(f"Could not analyze the image: {error}")
        else:
            with st.spinner("Predicting emotion..."):
                probabilities = model.predict(tensor, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index])

            left, right = st.columns([1, 2])
            with left:
                st.image(preview, caption=image_label, width="stretch")
            with right:
                st.metric("Prediction", class_names[predicted_index])
                st.metric("Confidence", f"{confidence:.1%}")
                probability_data = pd.DataFrame(
                    {
                        "Class": class_names,
                        "Probability": probabilities,
                    }
                ).sort_values("Probability", ascending=False)
                st.subheader("Class probability comparison")
                st.bar_chart(probability_data, x="Class", y="Probability")
                if class_names[predicted_index].startswith("unmapped output"):
                    st.warning(
                        "This output has no matching training label, so it "
                        "cannot be interpreted as a known emotion."
                    )

            if known_class_count < len(class_names):
                st.caption(
                    "The model has extra output classes without matching "
                    "training labels; predictions for those outputs are not "
                    "interpretable as emotions."
                )

elif mode == "Batch upload":
    uploads = st.file_uploader(
        "Choose multiple images",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key="batch_images",
    )
    if uploads:
        with st.spinner(f"Predicting {len(uploads)} image(s)..."):
            results, errors = predict_images(model, uploads, class_names)

        if not results.empty:
            average_confidence = results["Confidence"].mean()
            with st.container(horizontal=True):
                st.metric("Images predicted", len(results), border=True)
                st.metric("Average confidence", f"{average_confidence:.1%}", border=True)
                st.metric(
                    "Most common prediction",
                    results["Prediction"].mode().iloc[0],
                    border=True,
                )

            distribution = (
                results["Prediction"]
                .value_counts()
                .reindex(class_names, fill_value=0)
                .rename_axis("Class")
                .reset_index(name="Image count")
            )
            mean_probabilities = pd.DataFrame(
                {
                    "Class": class_names,
                    "Mean probability": [
                        results[f"Probability: {name}"].mean()
                        for name in class_names
                    ],
                }
            )

            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.subheader("Predicted class counts")
                st.bar_chart(distribution, x="Class", y="Image count")
            with chart_right:
                st.subheader("Average class probabilities")
                st.bar_chart(mean_probabilities, x="Class", y="Mean probability")

            st.subheader("Per-image predictions")
            st.dataframe(
                results.drop(
                    columns=[column for column in results if column.startswith("Probability:")]
                ),
                hide_index=True,
                width="stretch",
            )
            st.subheader("Per-image probability comparison")
            probability_columns = [
                f"Probability: {name}" for name in class_names
            ]
            chart_data = results.set_index("File")[probability_columns].rename(
                columns={
                    f"Probability: {name}": name
                    for name in class_names
                }
            )
            st.line_chart(chart_data)
            st.download_button(
                "Download predictions as CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="emotion_predictions.csv",
                mime="text/csv",
            )

        if errors:
            st.warning(f"{len(errors)} file(s) could not be processed.")
            st.dataframe(pd.DataFrame(errors), hide_index=True, width="stretch")

st.caption(
    "Predictions are model estimates, not a reliable assessment of a person's "
    "internal emotional state."
)
    