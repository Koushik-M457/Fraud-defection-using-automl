# Fraud-defection-using-automl
import numpy as np
import pandas as pd
df= pd.read_csv("/content/drive/MyDrive/dataset/bankfrauddefectiom.csv")
df.head()

d['Class'].value_counts()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X = d.drop('Class', axis=1)
y = d.Class
X = scalar.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# h2o.ai
import h2o
from h2o.automl import H2OAutoML
h2o.init()
df = h2o.import_file("/content/drive/MyDrive/dataset/bankfrauddefectiom.csv")
df.head(10)
df = h2o.import_file('/content/drive/MyDrive/dataset/bankfrauddefectiom.csv')
x = df.columns[:-1]  # Features
y = 'Class'  
train, test = df.split_frame(ratios=[0.8], seed=42)
aml = H2OAutoML(max_runtime_secs=300, seed=42)
aml.train(x=x, y=y, training_frame=train)
lb = aml.leaderboard
print(lb)

best_model = aml.leader
perf = best_model.model_performance(test)
print(perf)

best_model = aml.leader
perf = best_model.model_performance(test)
print(perf)

import seaborn as sns
sns.heatmap(confusion, annot=True)
from sklearn.metrics import classification_report , confusion_matrix
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is Fraud', 'is Normal'],columns=['predicted fraud','predicted normal'])
confusion
print(classification_report(y_test, y_predict))

#uding flaml
pip install flaml
from flaml import AutoML
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv("/content/drive/MyDrive/dataset/bankfrauddefectiom.csv")
df.head()
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

automl = AutoML()
automl_settings = {
    "time_budget": 600,  # in seconds
    "metric": 'accuracy',
    "task": 'classification',
    "log_file_name": 'credit_card_fraud_detection.log',
}
automl.fit(X_train, y_train, **automl_settings)
y_pred = automl.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
from sklearn.metrics import classification_report , confusion_matrix
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is Fraud', 'is Normal'],columns=['predicted fraud','predicted normal'])
confusion
print(classification_report(y_test, y_predict))
