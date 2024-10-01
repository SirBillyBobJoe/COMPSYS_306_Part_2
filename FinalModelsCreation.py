import pickle
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Load dataset
df = pickle.load(open('pickles/data32x32.pickle', 'rb'))
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Standardise data
scaler = StandardScaler()
x_train_normalised = scaler.fit_transform(x)
x_test_normalised = scaler.transform(x)
joblib.dump(scaler, 'joblibs/polyscaler32x32.joblib')

# Train SVM
print("Training...")
model = svm.SVC(
    C=0.01,
    kernel='poly',
    degree=3,
    gamma=0.1,
    probability=True,
    class_weight='balanced',
    max_iter=10000
)
model.fit(x_train_normalised, y)
print("Training completed")
# Save model
joblib.dump(model, 'joblibs/poly32x32-3.joblib')
print('SVM training completed.')

# Test Accuracy: 0.9831286073701346
# Best: 0.975133214920071 using {'C': 0.01, 'degree': 2, 'gamma': 0.1, 'kernel': 'poly'}
