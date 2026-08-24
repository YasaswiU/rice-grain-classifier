"""
app.py
------
Streamlit web application for the Rice Grain Classifier.

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from src import config
from src.predict import (
    predict_image,
    load_class_names,
    PredictionError,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Rice Grain Classifier",
    page_icon="🌾",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Model availability
# ---------------------------------------------------------------------------

cnn_available = os.path.exists(
    config.CNN_MODEL_PATH
)

vgg16_available = os.path.exists(
    config.VGG16_MODEL_PATH
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:

    st.header("🌾 About Project")

    st.write(
        "This application classifies images of rice grains "
        "into one of five rice varieties using deep learning."
    )

    st.write(
        "The final application uses VGG16 transfer learning "
        "because it achieved the best test performance."
    )

    # ---------------------------------------------------------------
    # Model information
    # ---------------------------------------------------------------

    st.header("🤖 Model Information")

    st.write(
        f"**VGG16:** "
        f"{'✅ Available' if vgg16_available else '❌ Not found'}"
    )

    st.write(
        f"**Custom CNN:** "
        f"{'Available' if cnn_available else 'Not found'}"
    )

    st.caption(
        "VGG16 Test Accuracy: 96.98%"
    )

    # ---------------------------------------------------------------
    # Dataset information
    # ---------------------------------------------------------------

    st.header("📊 Dataset Information")

    try:

        class_names = load_class_names()

        st.write(
            f"Detected classes ({len(class_names)}):"
        )

        for class_name in class_names:
            st.write(
                f"- {class_name}"
            )

    except Exception:

        st.warning(
            "Dataset information could not be loaded."
        )

    # ---------------------------------------------------------------
    # Technologies
    # ---------------------------------------------------------------

    st.header("🛠️ Technologies Used")

    st.write(
        "- TensorFlow / Keras\n"
        "- VGG16 Transfer Learning\n"
        "- Streamlit\n"
        "- scikit-learn\n"
        "- NumPy\n"
        "- Pandas\n"
        "- Matplotlib\n"
        "- Seaborn"
    )


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title(
    "🌾 Rice Grain Classifier"
)

st.subheader(
    "Deep Learning Based Rice Variety Classification"
)

st.write(
    "Upload an image of a rice grain and the trained VGG16 "
    "model will predict its rice variety along with the "
    "confidence score and probability distribution."
)


# ---------------------------------------------------------------------------
# Check VGG16 availability
# ---------------------------------------------------------------------------

if not vgg16_available:

    st.error(
        "The trained VGG16 model was not found."
    )

    st.write(
        "Expected model location:"
    )

    st.code(
        config.VGG16_MODEL_PATH
    )

    st.info(
        "Please make sure vgg16_best.keras exists inside "
        "the models folder."
    )

    st.stop()


# ---------------------------------------------------------------------------
# Final model
# ---------------------------------------------------------------------------

model_type = "vgg16"

st.success(
    "✅ VGG16 Transfer Learning Model Selected"
)

st.caption(
    "Test Accuracy: 96.98%"
)


# ---------------------------------------------------------------------------
# File uploader
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload a rice grain image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if uploaded_file is not None:

    col1, col2 = st.columns(
        2
    )

    # ================================================================
    # Uploaded image
    # ================================================================

    with col1:

        st.markdown(
            "### 📷 Uploaded Image"
        )

        try:

            image = Image.open(
                uploaded_file
            )

            # Convert to RGB so the model always receives
            # a 3-channel image.
            image = image.convert(
                "RGB"
            )

            st.image(
                image,
                use_container_width=True
            )

        except Exception:

            st.error(
                "This file could not be opened as an image. "
                "Please upload a valid JPG, JPEG, or PNG file."
            )

            st.stop()

    # ================================================================
    # Save temporary image
    # ================================================================

    temp_dir = os.path.join(
        config.BASE_DIR,
        ".streamlit_tmp"
    )

    os.makedirs(
        temp_dir,
        exist_ok=True
    )

    # Use a fixed temporary filename to avoid
    # problematic characters in uploaded filenames.
    temp_path = os.path.join(
        temp_dir,
        "uploaded_rice_image.jpg"
    )

    try:

        image.save(
            temp_path,
            format="JPEG"
        )

    except Exception as exc:

        st.error(
            f"Could not prepare the uploaded image: {exc}"
        )

        st.stop()

    # ================================================================
    # Prediction result
    # ================================================================

    with col2:

        st.markdown(
            "### 🔍 Prediction Result"
        )

        with st.spinner(
            "Analyzing rice grain image..."
        ):

            try:

                result = predict_image(
                    temp_path,
                    model_type=model_type
                )

            except PredictionError as exc:

                st.error(
                    str(exc)
                )

                result = None

            except Exception as exc:

                st.error(
                    "An unexpected error occurred "
                    f"while predicting: {exc}"
                )

                result = None

        # ------------------------------------------------------------
        # Display prediction
        # ------------------------------------------------------------

        if result is not None:

            predicted_class = result[
                "predicted_class"
            ]

            confidence = result[
                "confidence"
            ]

            probabilities = result[
                "probabilities"
            ]

            st.success(
                f"🌾 Predicted Rice Variety: "
                f"**{predicted_class}**"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            # --------------------------------------------------------
            # Confidence interpretation
            # --------------------------------------------------------

            if confidence >= 90:

                st.success(
                    "High-confidence prediction"
                )

            elif confidence >= 70:

                st.info(
                    "Moderate-confidence prediction"
                )

            else:

                st.warning(
                    "Low-confidence prediction"
                )

            # --------------------------------------------------------
            # Probability distribution
            # --------------------------------------------------------

            st.markdown(
                "### 📊 Probability Distribution"
            )

            prob_df = pd.DataFrame(
                {
                    "Variety": list(
                        probabilities.keys()
                    ),
                    "Probability (%)": list(
                        probabilities.values()
                    ),
                }
            )

            prob_df = prob_df.sort_values(
                "Probability (%)",
                ascending=False
            )

            # Bar chart
            st.bar_chart(
                prob_df.set_index(
                    "Variety"
                )
            )

            # Table
            st.dataframe(
                prob_df,
                use_container_width=True,
                hide_index=True
            )


    # ================================================================
    # Clean temporary file
    # ================================================================

    try:

        os.remove(
            temp_path
        )

    except OSError:

        pass

else:

    st.info(
        "👆 Upload a rice grain image above "
        "to get a prediction."
    )


# ---------------------------------------------------------------------------
# Project footer
# ---------------------------------------------------------------------------

st.divider()

st.caption(
    "Rice Grain Classifier — B.Tech Project | "
    "Built with TensorFlow, VGG16 Transfer Learning, "
    "and Streamlit."
)