import pandas as pd
from numpy import mean
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import xgboost
import numpy as np

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


#can add more models here
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
    "Histogram GB": HistGradientBoostingClassifier(random_state=42),
    "XG Boost" : xgboost.XGBClassifier(random_state=42)
}

# get best model
best_model = None
best_score = 0
best_model_name = None

for name, model in models.items():

    steps = [("preprocessor", preprocessor), 
                ("model", model)]
    pipeline = Pipeline(steps=steps)
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=-1)
    mean_score = mean(scores)
    
    print(f"Model: {name}, Mean ROC AUC: {mean_score:.3f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model = pipeline
        best_model_name = name

# best model
print(f"\nBest Model: {best_model_name} and ROC AUC: {best_score:.3f}")

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test)

# optional: get probability distribution of y
#import numpy as np
#import matplotlib.pyplot as plt

#plt.figure(figsize=(8, 5))
#plt.hist(y_prob, bins=20, color='skyblue', edgecolor='black')
#plt.title('Histogramm der vorhergesagten Wahrscheinlichkeiten')
#plt.xlabel('Wahrscheinlichkeit')
#plt.ylabel('Anzahl')
#plt.grid(True)
#plt.show()
#print(f"Test ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")

# set threshold because there will be no 1's otherwise
y_pred_threshold = (y_prob > np.quantile(y_prob, 0.95)).astype(int)
print("\nTest set evaluation")
print(classification_report(y_test, y_pred_threshold))

joblib.dump(best_model, "best_model.pkl")

print("Best model saved to 'best_model.pkl'.")
#optional: get ROC curve plot
#from sklearn.metrics import roc_curve, auc
#import matplotlib.pyplot as plt

# ROC-Kurve berechnen
#fpr, tpr, thresholds = roc_curve(y_test, y_prob)
#roc_auc = auc(fpr, tpr)

# Plot
#plt.figure(figsize=(8, 6))
#plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonale
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC-Kurve')
#plt.legend(loc='lower right')
#plt.grid(True)
#plt.show()


