# Hospital Re-admission

Hier verwende ich den Datensatz aus der Publikation "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records" von Strack et al. 2014 Link:https://onlinelibrary.wiley.com/doi/10.1155/2014/781670 

# Datenaufbereitung

In dem Notebook "preprocessing.ipynb" wird der Datensatz weiter aufbereitet. Dabei folge ich größenteils den Anweisungen aus Strack et al. 2014.
Es gibt allerdings kleine Abweichungen, die im Code durch Kommentare erklärt werden.

# Machine Learning Model

Das Model wird durch das Skript "ML_model.py" erzeugt und unter dem Namen "best_model.pkl" abgespeichert.
Die geringe relative Anzahl in der Ausprägung 1 in der binären abhängigen Variable wird durch "oversampling" ausgeglichen.
Es wird ein Random Forrest Classifier mit dem HistogramGradientClassifier aus sci-kit learn verglichen und das bessere Modell (gemessen anhand AUC) ausgegeben.

# Feature Relevanz

Durch Permutation wird berechnet, welche Variablen und welche Ausprägungen besonders relevant für die Klassifizierung sind.
Die Analyse zeigt, welche Ausprägungen in einem statistischen Zusammenhang mit einer Wiederaufnahme ins Krankenhaus stehen.
Der dazugehörige Code ist im Notebook "feature_relevance.ipynb" zu finden.    

