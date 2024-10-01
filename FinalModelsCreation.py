import pickle
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Load dataset
df = pickle.load(open('pickles/data100x100.pickle', 'rb'))
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Standardise data
scaler = StandardScaler()
x_train_normalised = scaler.fit_transform(x)
x_test_normalised = scaler.transform(x)
joblib.dump(scaler, 'joblibs/scaler100x100.joblib')

# Train SVM with linear kernel
print("Training...")
model = svm.SVC(
    C=0.001,
    kernel='linear',
    probability=True,
    class_weight='balanced',
    max_iter=10000
)
model.fit(x_train_normalised, y)
print("Training completed")
# Save model
joblib.dump(model, 'joblibs/standardised100x100.joblib')
print('SVM training completed.')
