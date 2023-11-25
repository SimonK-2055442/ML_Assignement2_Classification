import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = pd.read_csv("dataTitanic/titanic.csv")
print(data.info())


# delete or replace null values in row -> first try: delete
data.dropna(axis=0, subset=["age", "embarked", "fare"], inplace=True)
data.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)


# useful features are survived, sex, age, fare, sibsp (number of spouses, siblings on board), parch (number of parents, children on board), pclass (indication of social class -> 1 is highest, 3 lowest)
# drop the rest of the features for now
data.drop(columns=["name", "ticket", "cabin", "embarked", "boat", "body", "home.dest"], inplace=True)


# making a correlation matrix to see which features are most important
corr_matrix = data[["survived", "sex", "pclass", "parch", "fare"]].corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
plt.show()

# dataTitanic visualisation
fare_bins = [0, 20, 50, 100, 200, 1000]
fare_labels = ['0-20', '20-50', '50-100', '100-200', '200+']
fareGroup = pd.cut(data['fare'], bins=fare_bins, labels=fare_labels, right=False)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.barplot(x="sex", y="survived", data=data, ax=axes[0, 0])
axes[0, 0].set_title("Survival rate by sex")
sns.barplot(x="pclass", y="survived", data=data, ax=axes[0, 1])
axes[0, 1].set_title("Survival rate by pclass")
sns.barplot(x="parch", y="survived", data=data, ax=axes[1, 0])
axes[1, 0].set_title("Survival rate by parch")
sns.barplot(x=fareGroup, y="survived", data=data, ax=axes[1, 1])
axes[1, 1].set_title("Survival rate by fare")

plt.tight_layout()
plt.show()


features = ["sex", "age", "fare", "sibsp", "parch", "pclass"]
target = "survived"

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)


svm_classifier = SVC(kernel='linear', C=1)
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)

print("Support Vector Machine Classifier:")
print("Accuracy:", accuracy_score(y_test, svm_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_predictions))

print("\nDecision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test, dt_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_predictions))

