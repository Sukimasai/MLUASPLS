from __future__ import annotations
from pathlib import Path
import re
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Airline_review.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "airline_model.joblib"

NUMERIC_FEATURES = [
    "Overall_Rating",
    "Seat_Comfort",
    "Cabin_Staff_Service",
    "Food_and_Beverages",
    "Ground_Service",
    "Inflight_Entertainment",
    "Wifi_and_Connectivity",
    "Value_For_Money",
    "Verified",
]

CATEGORICAL_FEATURES = [
    "Type_Of_Traveller",
    "Seat_Type",
    "Aircraft",
    "Airline_Name",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

INSIGHT_RULES: dict[str, list[str]] = {
    "delay": ["delay", "late", "delayed", "cancel", "cancelled"],
    "staff": ["staff", "crew", "service"],
    "food": ["food", "meal", "snack", "drink", "beverage"],
    "seat": ["seat", "legroom", "space"],
    "baggage": ["baggage", "luggage", "bag", "bags"],
    "cleanliness": ["clean", "dirty", "hygiene"],
    "entertainment": ["movie", "entertainment", "ife", "wifi"],
    "pricing": ["price", "expensive", "value for money", "overpriced"],
}

def normalize_column_name(name: Any) -> str:
    text = str(name).strip()
    text = text.replace("&", "and")
    text = text.replace("/", " ")
    text = text.replace("-", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def normalize_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    dataframe = raw_df.copy()
    unnamed_columns = [column for column in dataframe.columns if str(column).startswith("Unnamed")]
    if unnamed_columns:
        dataframe = dataframe.drop(columns=unnamed_columns)
    dataframe.columns = [normalize_column_name(column) for column in dataframe.columns]
    return dataframe

def coerce_verified(series: pd.Series) -> pd.Series:
    text_values = series.astype(str).str.strip().str.lower()
    mapped = text_values.map({
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
    })
    mapped = mapped.where(series.notna(), np.nan)
    return mapped

def prepare_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    feature_frame = pd.DataFrame(index=dataframe.index)
    for column in NUMERIC_FEATURES:
        if column not in dataframe.columns:
            feature_frame[column] = pd.Series([np.nan] * len(dataframe), index=dataframe.index)
            continue
        if column == "Verified":
            feature_frame[column] = coerce_verified(dataframe[column])
        else:
            feature_frame[column] = pd.to_numeric(dataframe[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        if column not in dataframe.columns:
            feature_frame[column] = pd.Series(["Unknown"] * len(dataframe), index=dataframe.index)
            continue
        feature_frame[column] = dataframe[column].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
    return feature_frame[FEATURE_COLUMNS]

def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    classifier = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

def load_training_data(dataset_path: Path = DATASET_PATH) -> tuple[pd.DataFrame, pd.Series]:
    raw_df = pd.read_csv(dataset_path)
    normalized_df = normalize_dataframe(raw_df)
    if "Recommended" not in normalized_df.columns:
        raise ValueError("The dataset must contain a Recommended column.")
    normalized_df["Recommended"] = normalized_df["Recommended"].astype(str).str.strip().str.lower()
    normalized_df = normalized_df[normalized_df["Recommended"].isin({"yes", "no"})].copy()
    normalized_df["Recommended"] = normalized_df["Recommended"].map({"yes": 1, "no": 0}).astype(int)
    feature_frame = prepare_feature_frame(normalized_df)
    target = normalized_df["Recommended"]
    return feature_frame, target

def train_model(dataset_path: Path = DATASET_PATH, artifact_path: Path = MODEL_PATH) -> dict[str, Any]:
    features, target = load_training_data(dataset_path)
    pipeline = build_pipeline()
    pipeline.fit(features, target)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    joblib.dump(bundle, artifact_path)
    return bundle

def load_model_bundle(artifact_path: Path = MODEL_PATH) -> dict[str, Any]:
    if artifact_path.exists():
        bundle = joblib.load(artifact_path)
        if isinstance(bundle, Pipeline):
            bundle = {
                "model": bundle,
                "feature_columns": FEATURE_COLUMNS,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
            }
        return bundle
    return train_model(artifact_path=artifact_path)

def predict_recommendations(bundle: dict[str, Any], dataframe: pd.DataFrame) -> pd.DataFrame:
    model: Pipeline = bundle["model"]
    feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
    normalized_df = normalize_dataframe(dataframe)
    feature_frame = prepare_feature_frame(normalized_df)
    feature_frame = feature_frame.reindex(columns=feature_columns, fill_value=np.nan)
    predicted_labels = model.predict(feature_frame)
    predicted_probabilities = model.predict_proba(feature_frame)[:, 1]
    result = normalized_df.copy()
    result["predicted_recommended"] = predicted_labels
    result["predicted_recommended_prob"] = predicted_probabilities
    return result

def build_summary(predicted_frame: pd.DataFrame) -> dict[str, Any]:
    total_reviews = int(len(predicted_frame))
    recommended_reviews = int((predicted_frame["predicted_recommended"] == 1).sum())
    not_recommended_reviews = int((predicted_frame["predicted_recommended"] == 0).sum())
    average_confidence = float(predicted_frame["predicted_recommended_prob"].mean() * 100) if total_reviews else 0.0
    return {
        "total_reviews": total_reviews,
        "recommended_reviews": recommended_reviews,
        "not_recommended_reviews": not_recommended_reviews,
        "average_confidence": round(average_confidence, 1),
    }

def classify_priority(frequency: int, total_reviews: int) -> tuple[str, str, str]:
    percentage = (frequency / total_reviews) * 100
    if percentage > 20:
        return "Major Projects", "High", "Hard"
    if percentage > 10:
        return "Priority Target", "High", "Easy"
    if percentage < 5:
        return "Not Important", "Low", "Hard"
    return "Fill-ins", "Low", "Easy"

def analyze_insights(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    normalized_df = normalize_dataframe(dataframe)
    total_reviews = len(normalized_df)
    text_columns = [column for column in ["Review", "Review_Title", "Route", "Airline_Name"] if column in normalized_df.columns]
    if not text_columns:
        return []
    lower_text = normalized_df[text_columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    insights: list[dict[str, Any]] = []
    for category, keywords in INSIGHT_RULES.items():
        frequency = int(lower_text.apply(lambda value: any(keyword in value for keyword in keywords)).sum())
        if frequency == 0:
            continue
        priority, impact, effort = classify_priority(frequency, total_reviews)
        insights.append({
            "category": category,
            "issue": f"Customer feedback related to {category}",
            "frequency": frequency,
            "sentiment": "Mixed",
            "priority": priority,
            "impact": impact,
            "effort": effort,
        })
    insights.sort(key=lambda item: item["frequency"], reverse=True)
    return insights

def build_priority_matrix(insights: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    matrix = {
        "priority_target": [],
        "major_projects": [],
        "fill_ins": [],
        "not_important": [],
    }
    for insight in insights:
        item = {"category": insight["category"], "frequency": insight["frequency"]}
        if insight["priority"] == "Priority Target":
            matrix["priority_target"].append(item)
        elif insight["priority"] == "Major Projects":
            matrix["major_projects"].append(item)
        elif insight["priority"] == "Fill-ins":
            matrix["fill_ins"].append(item)
        else:
            matrix["not_important"].append(item)
    return matrix

def build_dashboard_payload(dataframe: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    predicted_frame = predict_recommendations(bundle, dataframe)
    summary = build_summary(predicted_frame)
    insights = analyze_insights(dataframe)
    matrix = build_priority_matrix(insights)
    return {"summary": summary, "insights": insights, "matrix": matrix}