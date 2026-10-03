"""
FEATURE SPECIFICATION — the single source of truth for every input feature.

WHAT: Describes each of the 8 inputs: API name, CSV column, label, unit,
      valid range, whether it is required, and the user-facing error message.
WHY:  The backend validation (Pydantic), the training code (column names,
      zero-as-missing rule) and the React form (fetched via GET /feature-info)
      all read from THIS file, so the three can never disagree.
VIVA: "Validation limits are defined once in features.py. The backend enforces
      them with Pydantic and the frontend downloads them from /feature-info,
      so the client and server always use identical rules."

About the valid ranges:
  They are broad, physiologically plausible limits used to reject impossible
  input (e.g. negative age, BMI of 500). They are NOT the training-data
  ranges. Values that are valid but outside what the model saw in training
  are accepted and flagged with an out-of-distribution warning instead.
"""
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class FeatureSpec:
    key: str                  # name used in the API / JSON
    csv_column: str           # column name in diabetes.csv
    label: str                # label shown in the form
    unit: str                 # unit shown next to the field
    min_value: float          # smallest accepted value
    max_value: float          # largest accepted value
    step: float               # input step for the HTML number field
    integer: bool             # must be a whole number?
    required: bool            # False -> may be left blank (imputed by the pipeline)
    zero_means_missing: bool  # in the raw dataset, 0 encodes "not measured"
    placeholder: str
    description: str
    error_message: str


FEATURES: list[FeatureSpec] = [
    FeatureSpec(
        key="pregnancies", csv_column="Pregnancies",
        label="Pregnancies", unit="count",
        min_value=0, max_value=20, step=1, integer=True, required=True,
        zero_means_missing=False,
        placeholder="e.g. 2",
        description="Number of times pregnant.",
        error_message="Please enter a valid number of pregnancies (whole number, 0–20).",
    ),
    FeatureSpec(
        key="glucose", csv_column="Glucose",
        label="Glucose", unit="mg/dL",
        min_value=40, max_value=400, step=1, integer=False, required=True,
        zero_means_missing=True,
        placeholder="e.g. 120",
        description="Plasma glucose concentration 2 hours after an oral glucose tolerance test.",
        error_message="Please enter a valid glucose level (40–400 mg/dL).",
    ),
    FeatureSpec(
        key="blood_pressure", csv_column="BloodPressure",
        label="Blood Pressure (diastolic)", unit="mmHg",
        min_value=20, max_value=140, step=1, integer=False, required=True,
        zero_means_missing=True,
        placeholder="e.g. 70",
        description="Diastolic blood pressure.",
        error_message="Please enter a valid diastolic blood pressure (20–140 mmHg).",
    ),
    FeatureSpec(
        key="skin_thickness", csv_column="SkinThickness",
        label="Skin Thickness (triceps)", unit="mm",
        min_value=5, max_value=100, step=1, integer=False, required=False,
        zero_means_missing=True,
        placeholder="Optional, e.g. 20",
        description="Triceps skin-fold thickness. Optional: if left blank, the model's imputer fills it.",
        error_message="Please enter a valid skin thickness (5–100 mm) or leave it blank.",
    ),
    FeatureSpec(
        key="insulin", csv_column="Insulin",
        label="Insulin (2-hour serum)", unit="µU/mL",
        min_value=10, max_value=900, step=1, integer=False, required=False,
        zero_means_missing=True,
        placeholder="Optional, e.g. 80",
        description="2-hour serum insulin. Optional: if left blank, the model's imputer fills it.",
        error_message="Please enter a valid insulin value (10–900 µU/mL) or leave it blank.",
    ),
    FeatureSpec(
        key="bmi", csv_column="BMI",
        label="BMI", unit="kg/m²",
        min_value=12, max_value=70, step=0.1, integer=False, required=True,
        zero_means_missing=True,
        placeholder="e.g. 28.5",
        description="Body mass index (weight in kg / height in m²).",
        error_message="Please enter a valid BMI (12–70 kg/m²).",
    ),
    FeatureSpec(
        key="diabetes_pedigree", csv_column="DiabetesPedigreeFunction",
        label="Diabetes Pedigree Function", unit="score",
        min_value=0.05, max_value=3.0, step=0.001, integer=False, required=True,
        zero_means_missing=False,
        placeholder="e.g. 0.35",
        description="Score summarising family history of diabetes.",
        error_message="Please enter a valid diabetes pedigree value (0.05–3.0).",
    ),
    FeatureSpec(
        key="age", csv_column="Age",
        label="Age", unit="years",
        min_value=21, max_value=100, step=1, integer=True, required=True,
        zero_means_missing=False,
        placeholder="e.g. 35",
        description="Age in years. The training data only contains adults aged 21+.",
        error_message="Please enter a valid age (whole number, 21–100 years).",
    ),
]

TARGET_COLUMN = "Outcome"

# Handy lookups
FEATURE_KEYS: list[str] = [f.key for f in FEATURES]
CSV_COLUMNS: list[str] = [f.csv_column for f in FEATURES]
SPEC_BY_KEY: dict[str, FeatureSpec] = {f.key: f for f in FEATURES}
KEY_TO_CSV: dict[str, str] = {f.key: f.csv_column for f in FEATURES}
CSV_TO_KEY: dict[str, str] = {f.csv_column: f.key for f in FEATURES}
ZERO_AS_MISSING_COLUMNS: list[str] = [f.csv_column for f in FEATURES if f.zero_means_missing]


def feature_as_dict(spec: FeatureSpec, training_range: Optional[dict] = None) -> dict:
    """Serialise one feature spec for the /feature-info endpoint."""
    data = asdict(spec)
    data["training_range"] = training_range
    return data
