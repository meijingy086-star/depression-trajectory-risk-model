import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgba
from pathlib import Path
import hashlib
import logging

from presentation import (class_display_names, input_display_name,
                          category_display_labels, UNITS)

from predictor import TrajectoryModel


st.set_page_config(
    page_title="Depressive Symptom Trajectory Predictor",
    page_icon="🧠",
    layout="centered",
)


# Include the artifact fingerprint in the cache key after model replacement.
@st.cache_resource
def load_model(path, fingerprint):
    return TrajectoryModel.load(path)


artifact_path = Path(__file__).absolute().parent / "artifacts" / "model.json.gz"
if not artifact_path.is_file():
    logging.getLogger(__name__).error("Required prediction artifact is missing: %s", artifact_path)
    st.error("The prediction model is temporarily unavailable. Please contact the app owner.")
    st.stop()
artifact_fingerprint = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
model = load_model(str(artifact_path), artifact_fingerprint)

# Class numbers have no intrinsic trajectory meaning after refitting.
group_names = class_display_names(model.data, model.classes)

variable_names = [spec["name"] for spec in model.inputs]


feature_names = [input_display_name(spec) for spec in model.inputs]


st.title("Depressive Symptom Trajectory Predictor")


def cognition_calculator():
    with st.sidebar:
        with st.expander("Cognition score calculator (0–21)"):
            with st.form("cognition_assessment"):
                st.markdown("**1. Immediate recall**")
                st.write("Read these ten words aloud. Ask the participant to recall as many as possible in any order.")
                st.caption("Vegetables, hat, train, land, river, table, apple, gift, sky, scissors")
                immediate = st.number_input("Number of distinct words recalled correctly now", 0, 10, value=None)

                st.markdown("**2. Time orientation**")
                orientation = [
                    st.selectbox(question, ("Correct", "Incorrect / Don't know"), index=None)
                    for question in (
                        "What year is it?", "What month is it?",
                        "What is today's date (day of the month)?",
                        "What day of the week is it?", "What season is it?",
                    )
                ]

                st.markdown("**3. Serial sevens**")
                st.write("Ask the participant to start at 100 and subtract 7 five times. Record each answer as given. Enter ? for a missed answer.")
                responses = [st.text_input(f"Answer {i}") for i in range(1, 6)]

                st.markdown("**4. Figure copying**")
                st.write("Show the reference drawing of two intersecting pentagons and ask the participant to copy it.")
                drawing = st.selectbox(
                    "Did the participant draw two recognizable pentagons that intersect?",
                    ("Yes", "No"), index=None,
                )

                st.markdown("**5. Delayed recall**")
                st.write("After the intervening tasks (about 4–10 minutes), ask for the same words again without rereading the list.")
                delayed = st.number_input("Number of distinct words recalled correctly after the delay", 0, 10, value=None)
                calculate = st.form_submit_button("Calculate and use score")

            if calculate:
                if immediate is None or delayed is None or None in orientation or drawing is None:
                    st.error("Complete both recall counts, all five orientation questions, and the drawing assessment.")
                else:
                    try:
                        answers = [None if answer.strip() == "?" else int(answer) for answer in responses]
                    except ValueError:
                        st.error("Enter a whole number or ? for each serial subtraction answer.")
                    else:
                        memory = (immediate + delayed) / 2
                        serial = sum(
                            previous is not None and current == previous - 7
                            for previous, current in zip([100] + answers[:-1], answers)
                        )
                        mental = orientation.count("Correct") + serial + (drawing == "Yes")
                        total = memory + mental
                        st.session_state["cognition_result"] = (memory, mental, total)
                        st.session_state["total_cognition_input"] = total

            if "cognition_result" in st.session_state:
                memory, mental, total = st.session_state["cognition_result"]
                st.success(f"Calculated total cognition score: {total:g} / 21")
                st.caption(f"Episodic memory: {memory:g} / 10 · Mental intactness: {mental} / 11")


cognition_calculator()


# Generate input fields
inputs = {}
input_columns = st.columns(2)

for input_index, spec in enumerate(model.inputs):
    name = spec["name"]
    label = input_display_name(spec)
    container = input_columns[input_index % 2]

    if spec["type"] == "category":
        options = [item["value"] for item in spec["options"]]
        labels = category_display_labels(spec)

        with container:
            inputs[name] = st.selectbox(
                label,
                options=options,
                index=options.index(str(spec["default"])),
                format_func=lambda value, mapping=labels: mapping[str(value)],
                help=spec.get("help_en"),
            )

    else:
        unit = UNITS.get(name, "")
        display_label = f"{label} ({unit})" if unit else label
        number_options = {}
        default_value = float(spec["default"])
        if name == "total_cognition":
            display_label = f"{label} ({unit}; range: 0–21)"
            st.session_state.setdefault("total_cognition_input", default_value)
            default_value = None
            number_options = {
                "min_value": 0.0,
                "max_value": 21.0,
                "key": "total_cognition_input",
            }

        with container:
            inputs[name] = st.number_input(
                display_label,
                value=default_value,
                step=0.1,
                format="%.2f",
                help=spec.get("help_en"),
                **number_options,
            )


# Convert SHAP inputs back to the original model inputs
def predict_proba(values):
    values = np.asarray(values, dtype=float)
    probabilities = []

    for row in values:
        raw = {}

        for i, spec in enumerate(model.inputs):
            if spec["type"] == "category":
                raw[spec["name"]] = str(int(row[i]))
            else:
                raw[spec["name"]] = float(row[i])

        probabilities.append(model.predict(raw)["probabilities"])

    return np.asarray(probabilities)


def style_shap_force_plot(fig):
    """Match the R force plot: positive yellow, negative rose-purple."""
    palette = {
        "#ff0d57": "#F6D746",
        "#ff0051": "#F6D746",
        "#1e88e5": "#A32B63",
        "#ffc3d5": "#FDF5D1",
        "#d1e6fa": "#E8CAD8",
    }

    def recolor(color):
        rgba = to_rgba(color)
        replacement = palette.get(to_hex(rgba))
        return to_rgba(replacement, alpha=rgba[3]) if replacement else rgba

    # SHAP's Matplotlib renderer uses fixed colors; style only this figure.
    for ax in fig.axes:
        for patch in ax.patches:
            patch.set_facecolor(recolor(patch.get_facecolor()))
            patch.set_edgecolor(recolor(patch.get_edgecolor()))
        for line in ax.lines:
            line.set_color(recolor(line.get_color()))
        for text in ax.texts:
            text.set_color(recolor(text.get_color()))
        for image in ax.images:
            color = palette.get(to_hex(image.get_cmap()(0.0)))
            if color:
                image.set_cmap(
                    LinearSegmentedColormap.from_list(
                        "trajectory_shap", [color, "#FFFFFF"]
                    )
                )


if st.button("Predict trajectory group", type="primary", use_container_width=True):
    if inputs.get("total_cognition") is None:
        st.error("Enter a total cognition score or calculate it in the sidebar.")
        st.stop()
    prediction = model.predict(inputs)

    predicted_group = group_names[prediction["class"]]
    st.success("Predicted trajectory group: **{}**".format(predicted_group))

    features = pd.DataFrame(
        [[float(inputs[name]) for name in variable_names]],
        columns=feature_names,
    )

    # Reference profile: development-set medians and modes
    background = pd.DataFrame(
        [[float(spec["default"]) for spec in model.inputs]],
        columns=feature_names,
    )

    with st.spinner("Calculating SHAP values..."):
        explainer = shap.PermutationExplainer(
            predict_proba,
            background,
            feature_names=feature_names,
            seed=2026,
        )

        explanation = explainer(
            features,
            max_evals=32 * (2 * len(feature_names) + 1),
        )

    # Explain the predicted class
    class_index = model.classes.index(prediction["class"])
    shap_values = explanation.values[0, :, class_index]
    base_value = float(explanation.base_values[0, class_index])

    st.subheader("Individual SHAP Analysis")

    if np.allclose(shap_values, 0):
        st.write(
            "All SHAP values are zero relative to the reference profile."
        )

    else:
        fig = shap.force_plot(
            base_value,
            shap_values,
            features.iloc[0],
            feature_names=feature_names,
            out_names=predicted_group,
            matplotlib=True,
            show=False,
            figsize=(16, 4),
        )

        style_shap_force_plot(fig)
        st.pyplot(fig)
        plt.close(fig)

st.caption(
    "ADL impairment: difficulty in at least one of dressing, bathing, eating, getting in or "
    "out of bed, toileting, or continence. IADL impairment: difficulty in at least one of "
    "housework, cooking, shopping, taking medication, or managing money; the 2011 baseline "
    "definition excludes telephone use."
)


