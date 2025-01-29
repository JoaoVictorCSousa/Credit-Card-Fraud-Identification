import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

#Displaying Dataset
data = pd.read_csv("creditcard.csv")
data

#Displaying detailed dataset informations
print(data.info())
print(data["Class"].value_counts())  #Displays the distribution of fraud (Class 1) and non-fraud (Class 0)

#Class filtering and balancing using undersampling
fraud = data[data["Class"] == 1]  # Fraudulent transactions
non_fraud = data[data["Class"] == 0].sample(len(fraud))  #Sample of non-fraudulent transactions

#Creation of the balanced dataset and separation of dependent variables from independent ones
balanced_data = pd.concat([fraud, non_fraud])
X = balanced_data.drop("Class", axis=1)
y = balanced_data["Class"]

#Division into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Decision Tree Model
tree_model = DecisionTreeClassifier(random_state=42, max_depth=3)

#Training
tree_model.fit(X_train, y_train)

#Prediction
y_pred = tree_model.predict(X_test)

#Confusion Matrix

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Results report
print("\nClassification report:")
print(classification_report(y_test, y_pred))

#Accuracy
print("\Accuracy:", accuracy_score(y_test, y_pred))


#Decision Tree Figure
plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=X.columns, class_names=["Non-Fraud", "Fraud"], filled=True, fontsize=10)
plt.show()

