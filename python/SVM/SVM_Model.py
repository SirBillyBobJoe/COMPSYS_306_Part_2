import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report

with open('./pickles/data.pickle', 'rb') as f:
    df = pickle.load(f)

X=df.drop(columns=['Target'])
y=df['Target']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=77, stratify=y)

param_grid = [
    {
        'C': [0.001 ],
        'kernel': ['linear'], 
    }
]


svc=svm.SVC(probability=True, class_weight='balanced',max_iter=10000)
model = GridSearchCV(svc, param_grid, cv=3, scoring='accuracy', verbose=3)
model.fit(X_train,y_train)

best_model = model.best_estimator_
print(f"Best Parameters: {model.best_params_}")

with open('./pickles/SVM_Model.pickle', 'wb') as f:
    pickle.dump(best_model, f)

with open('./pickles/SVM_Model.pickle', 'rb') as f:
    model = pickle.load(f)

y_pred=model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)