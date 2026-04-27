import warnings
warnings.filterwarnings("ignore")

import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


INPUT_FILE = "mark_data.csv"
MIN_GENRE_COUNT = 15
TOP_N_GENRES_IN_CM = 15
TOP_N_DESCRIPTOR_IMPORTANCE = 25
RANDOM_STATE = 42


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Expects a TSV where:
    - first column = Genre
    - remaining columns = descriptor tokens
    - rows may have variable numbers of descriptor columns
    """
    rows = []

    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            # Strip whitespace from each field
            row = [cell.strip() for cell in row]

            # Skip blank lines
            if not any(cell != "" for cell in row):
                continue

            rows.append(row)

    if not rows:
        raise ValueError("The input file is empty or unreadable.")

    # Pad rows to equal length
    max_len = max(len(row) for row in rows)
    padded_rows = [row + [""] * (max_len - len(row)) for row in rows]

    df = pd.DataFrame(padded_rows)

    if df.shape[1] < 2:
        raise ValueError(
            "The file must have at least 2 columns: Genre and at least one descriptor."
        )

    df = df.rename(columns={0: "Genre"})
    descriptor_cols = [c for c in df.columns if c != "Genre"]

    df["Genre"] = df["Genre"].fillna("").astype(str).str.strip()

    for col in descriptor_cols:
        df[col] = df[col].fillna("").astype(str).str.strip().str.lower()

    # Combine descriptor columns into one text field
    df["Descriptors"] = df[descriptor_cols].apply(
        lambda row: " ".join(token for token in row if token != ""),
        axis=1,
    )

    df = df[["Genre", "Descriptors"]].copy()

    # Drop rows missing genre or descriptors
    df = df[(df["Genre"] != "") & (df["Descriptors"] != "")].copy()

    # Remove duplicate descriptor tokens within a row, preserving order
    def dedupe_tokens(text: str) -> str:
        seen = set()
        ordered = []
        for tok in text.split():
            if tok not in seen:
                seen.add(tok)
                ordered.append(tok)
        return " ".join(ordered)

    df["Descriptors"] = df["Descriptors"].apply(dedupe_tokens)

    return df.reset_index(drop=True)


def filter_rare_genres(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    genre_counts = df["Genre"].value_counts()
    keep_genres = genre_counts[genre_counts >= min_count].index
    filtered = df[df["Genre"].isin(keep_genres)].copy()
    return filtered.reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame):
    vectorizer = CountVectorizer(binary=True)
    X_desc = vectorizer.fit_transform(df["Descriptors"])
    return X_desc, vectorizer


def evaluate_model(name, model, X_test, y_test, label_encoder):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted", zero_division=0
    )

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            preds,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Predictions": preds,
    }


def save_confusion_matrix(y_test, preds, label_encoder, output_path, top_n=15):
    y_test_series = pd.Series(y_test)
    top_class_ids = y_test_series.value_counts().head(top_n).index.tolist()

    mask = np.isin(y_test, top_class_ids)
    y_test_top = y_test[mask]
    preds_top = preds[mask]

    labels = sorted(top_class_ids)
    cm = confusion_matrix(y_test_top, preds_top, labels=labels)

    class_names = [label_encoder.inverse_transform([i])[0] for i in labels]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix (Top Genres)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_genre_distribution(df, output_path):
    genre_counts = df["Genre"].value_counts().head(20)

    plt.figure(figsize=(12, 7))
    plt.bar(genre_counts.index, genre_counts.values)
    plt.title("Top 20 Genre Counts")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_logistic_descriptor_importance(model, vectorizer, output_path, top_n=25):
    if not hasattr(model, "coef_"):
        return

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = np.abs(model.coef_)
    avg_importance = coefs.mean(axis=0)

    importance_df = pd.DataFrame({
        "Descriptor": feature_names,
        "Importance": avg_importance
    }).sort_values("Importance", ascending=False)

    importance_df.to_csv(output_path, index=False)

    top_df = importance_df.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["Descriptor"][::-1], top_df["Importance"][::-1])
    plt.title("Top Descriptor Importance (Logistic Regression)")
    plt.xlabel("Average Absolute Coefficient")
    plt.tight_layout()
    plt.savefig("top_descriptor_importance.png", dpi=300)
    plt.close()


def main():
    print("Loading dataset...")
    df = load_dataset(INPUT_FILE)

    print(f"Initial rows: {len(df)}")
    print(f"Initial unique genres: {df['Genre'].nunique()}")

    df = filter_rare_genres(df, MIN_GENRE_COUNT)

    print(f"Rows after filtering rare genres (< {MIN_GENRE_COUNT}): {len(df)}")
    print(f"Remaining unique genres: {df['Genre'].nunique()}")

    if len(df) < 20:
        raise ValueError("Too few rows remain after filtering. Lower MIN_GENRE_COUNT.")

    save_genre_distribution(df, "top_genres_distribution.png")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Genre"])

    X, vectorizer = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\nTraining Logistic Regression...")
    log_reg_grid = {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    }
    log_reg_search = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE),
        log_reg_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
        verbose=1,
    )
    log_reg_search.fit(X_train, y_train)
    best_log_reg = log_reg_search.best_estimator_
    print("Best Logistic Regression params:", log_reg_search.best_params_)

    print("\nTraining Decision Tree...")
    tree_grid = {
        "max_depth": [10, 20, 40, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    tree_search = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        tree_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    tree_search.fit(X_train, y_train)
    best_tree = tree_search.best_estimator_
    print("Best Decision Tree params:", tree_search.best_params_)

    print("\nTraining k-NN...")
    knn_grid = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski"],
    }
    knn_search = GridSearchCV(
        KNeighborsClassifier(),
        knn_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    knn_search.fit(X_train, y_train)
    best_knn = knn_search.best_estimator_
    print("Best k-NN params:", knn_search.best_params_)

    results = []
    model_objects = {
        "Logistic Regression": best_log_reg,
        "Decision Tree": best_tree,
        "k-NN": best_knn,
    }

    for name, model in model_objects.items():
        result = evaluate_model(name, model, X_test, y_test, label_encoder)
        results.append({
            "Model": result["Model"],
            "Accuracy": result["Accuracy"],
            "Precision": result["Precision"],
            "Recall": result["Recall"],
            "F1": result["F1"],
        })

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    results_df.to_csv("genre_model_results.csv", index=False)

    print("\n=== Model Comparison ===")
    print(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = model_objects[best_model_name]
    best_preds = best_model.predict(X_test)

    save_confusion_matrix(
        y_test,
        best_preds,
        label_encoder,
        "best_genre_model_confusion_matrix.png",
        top_n=TOP_N_GENRES_IN_CM,
    )

    save_logistic_descriptor_importance(
        best_log_reg,
        vectorizer,
        "genre_descriptor_importance.csv",
        top_n=TOP_N_DESCRIPTOR_IMPORTANCE,
    )

    print("\nFiles created:")
    print("- genre_model_results.csv")
    print("- genre_descriptor_importance.csv")
    print("- best_genre_model_confusion_matrix.png")
    print("- top_genres_distribution.png")
    print("- top_descriptor_importance.png")
    print("\nDone.")


if __name__ == "__main__":
    main()