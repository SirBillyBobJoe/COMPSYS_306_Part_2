import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Load dataset
data_dict = pickle.load(open('./pickles/data.pickle', 'rb'))
# Extract features and target from the dictionary
x = data_dict['data']
y = data_dict['target']

# Load dataset
data_dict = pickle.load(open('./pickles/data.pickle', 'rb'))
# Extract features and target from the dictionary
x = data_dict['data']
y = data_dict['target']

# Standardize data
scaler = StandardScaler()
x_train_normalised = scaler.fit_transform(x)
x_test_normalised = scaler.transform(x)
joblib.dump(scaler, './joblibs/scaler.joblib')

    # Train SVM
print("Training...")
model = svm.SVC(
    C=0.001,
    kernel='poly',
    degree=2,
    gamma=0.1,
    probability=True,
    class_weight='balanced',
    max_iter=10000
)
model.fit(x_train_normalised, y)
print("Training completed")

joblib.dump(model, 'joblibs/svm_model.joblib')
print('SVM training completed.')
