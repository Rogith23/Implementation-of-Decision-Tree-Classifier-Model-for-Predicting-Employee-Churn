# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Collection

Collect or import the dataset that contains employee information.

Example features: satisfaction_level, last_evaluation, number_project, average_montly_hours, time_spend_company, etc.

2. Data Preprocessing

Handle missing values (if any).

Encode categorical variables (e.g., department, salary) using Label Encoding or One-Hot Encoding.

Feature scaling (not strictly needed for Decision Tree but may help in general pipelines).

3. Train-Test Split

Split the dataset into training and testing sets (typically 70-30 or 80-20).

4. Model Training

Initialize and train the Decision Tree Classifier.

5. Model Prediction

Predict churn on the test set.

6. Model Evaluation

Evaluate the model using metrics:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

7. (Optional) Model Visualization

Visualize the decision tree to interpret decisions.

8. Model Tuning

Tune hyperparameters like:

max_depth

min_samples_split

min_samples_leaf

## Program:

```
import pandas as pd
df=pd.read_csv("Employee.csv")
df.head
print(df.info)
df.isnull().sum()
df["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=df["left"]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeCl(assifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
confusion=metrics.confusion_matrix(y_test,y_pred)
classification=metrics.classification_report(y_test,y_pred)
print("Name:Rogith.J")
print("Reg no:212224040280")
print("Accuracy:", accuracy)
print("Confusion Matrix:", confusion)
print("Classification Report:", classification)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![decision tree classifier model](sam.png)
data.head():


<img width="1123" height="644" alt="Screenshot 2025-09-22 131527" src="https://github.com/user-attachments/assets/9ee71d4e-dfdc-4dd9-81d9-cf01158f8ab1" />


data.info():


<img width="1095" height="818" alt="Screenshot 2025-09-22 131540" src="https://github.com/user-attachments/assets/ba8bdc23-81c0-40ac-88a3-b86078b7a95f" />

isnull() and sum():

<img width="658" height="294" alt="Screenshot 2025-09-22 131549" src="https://github.com/user-attachments/assets/2e074fe0-d3a7-4e6a-bd79-d2e021207428" />

Data Value Counts():


<img width="937" height="149" alt="Screenshot 2025-09-22 131605" src="https://github.com/user-attachments/assets/05cd5315-02f9-46c4-9200-6b1ab5e0d20c" />

Data.head() for salary:

<img width="1354" height="304" alt="Screenshot 2025-09-22 131615" src="https://github.com/user-attachments/assets/9d826041-4659-46cc-a22b-794f70463730" />

x.head:

<img width="1377" height="310" alt="Screenshot 2025-09-22 131625" src="https://github.com/user-attachments/assets/0b4781ae-36ef-4830-839d-de6777f9465e" />

Accuracy Value:


<img width="894" height="327" alt="Screenshot 2025-09-22 131647" src="https://github.com/user-attachments/assets/8e0cc4ea-48a0-4f85-9e78-9c8d280ff117" />

Data Prediction:


<img width="1371" height="185" alt="Screenshot 2025-09-22 131653" src="https://github.com/user-attachments/assets/d93db4a9-ce2d-4256-8636-8f5f9b912fba" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
