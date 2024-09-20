import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
import joblib

with open("./pickles/data.pickle", "rb") as f:
    df = pickle.load(f)

X = df.drop(columns=["Target"])
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=77, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

joblib.dump(scaler, "./joblibs/scaler.joblib")

param_grid = [
    {
        "C": [0.01],
        "kernel": ["linear"],
    },
        {
        "C": [0.01],
        "kernel": ["rbf"],
        "gamma":[0.1,1],
    },
        {
        "C": [0.01],
        "kernel": ["linear"],
        "gamma":[0.1,1],
        "degree":[3,4]
    }
]


svc = svm.SVC(probability=True, class_weight="balanced", max_iter=10000)
model = GridSearchCV(svc, param_grid, cv=3, scoring="accuracy", verbose=3)
model.fit(X_train, y_train)

best_model = model.best_estimator_
print(f"Best Parameters: {model.best_params_}")

# Save the best model using joblib
joblib.dump(best_model, "./joblibs/SVM_Model.joblib")

# Load the saved model using joblib
model = joblib.load("./joblibs/SVM_Model.joblib")

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

