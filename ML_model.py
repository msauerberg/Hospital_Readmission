import pandas as pd
from numpy import mean
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import joblib

final_dta = pd.read_csv("data_preprocessed.csv")

X = final_dta.drop(columns="target")
y = final_dta["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [col for col in X.columns if col not in num_cols]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# for over- and under-sampling, I am following this example:
#https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
k_values = [1, 2, 3, 4, 5, 6, 7]

#can add more models here
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Histogram GB": HistGradientBoostingClassifier(random_state=42)
}

# get best model
best_model = None
best_score = 0
best_k = None
best_model_name = None

for k in k_values:
    for name, model in models.items():
        over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
        under = RandomUnderSampler(sampling_strategy=0.5)
    
        steps = [("preprocessor", preprocessor), 
                 ("over", over), 
                 ("under", under), 
                 ("model", model)]
        pipeline = Pipeline(steps=steps)
        
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
        mean_score = mean(scores)
        
        print(f"Model: {name}, k={k}, Mean ROC AUC: {mean_score:.3f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = pipeline
            best_k = k
            best_model_name = name

# best model
print(f"\nBest Model: {best_model_name} with k={best_k} and ROC AUC: {best_score:.3f}")

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test)

print("\nTest set evaluation")
print(classification_report(y_test, y_pred))
print(f"Test ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")

joblib.dump(best_model, "best_model.pkl")

print("Best model saved to 'best_model.pkl'.")