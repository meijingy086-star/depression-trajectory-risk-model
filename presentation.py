"""Display names for model-defined inputs and explicitly verified classes."""

INPUT_LABELS = {
    "gender": "Gender", "rural": "Residence", "srh": "Self-rated health",
    "adlab_c": "ADL impairment", "arthre": "Arthritis",
    "digeste": "Stomach or digestive disease", "retire": "Retirement status",
    "satlife": "Life satisfaction", "pain": "Pain", "sleep": "Sleep duration",
    "disability": "Disability", "fall_down": "History of falls",
    "eyesight_close": "Near vision", "eyesight_distance": "Distance vision",
    "hear": "Hearing", "total_cognition": "Total cognition score",
    "pension": "Pension insurance", "iadl": "IADL impairment",
}

CATEGORY_LABELS = {
    "gender": {"0": "Female", "1": "Male"},
    "rural": {"0": "Urban", "1": "Rural"},
    "srh": {"1": "Very poor", "2": "Poor", "3": "Fair", "4": "Good", "5": "Very good"},
    "satlife": {"1": "Not at all satisfied", "2": "Not very satisfied",
                "3": "Somewhat satisfied", "4": "Very satisfied", "5": "Completely satisfied"},
    "hear": {"1": "Poor", "2": "Fair", "3": "Good", "4": "Very good", "5": "Excellent"},
    "eyesight_close": {"1": "Poor", "2": "Fair", "3": "Good", "4": "Very good", "5": "Excellent"},
    "eyesight_distance": {"1": "Poor", "2": "Fair", "3": "Good", "4": "Very good", "5": "Excellent"},
    "adlab_c": {"0": "No impairment", "1": "Impairment"},
    "iadl": {"0": "No impairment", "1": "Impairment"},
}
BINARY_VARIABLES = {"arthre", "digeste", "retire", "pain", "disability",
                    "fall_down", "pension"}
UNITS = {"sleep": "hours/night", "total_cognition": "points"}


def class_display_names(model_data, classes):
    configured = model_data.get("class_labels", {})
    if not isinstance(configured, dict):
        configured = {}
    return {name: configured.get(name) or name.replace("Class", "Class ")
            for name in classes}


def input_display_name(spec):
    return spec.get("label_en") or INPUT_LABELS.get(spec["name"], spec["name"].replace("_", " ").title())


def category_display_labels(spec):
    name = spec["name"]
    known = CATEGORY_LABELS.get(name, {"1": "No", "2": "Yes"} if name in BINARY_VARIABLES else {})
    return {str(option["value"]): option.get("label_en") or
            known.get(str(option["value"]), "Code " + str(option["value"]))
            for option in spec["options"]}
