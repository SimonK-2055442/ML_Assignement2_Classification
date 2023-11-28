import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = pd.read_csv("dataTitanic/titanic.csv")
print(data.info())

# useful features are survived, sex, age, fare, sibsp (number of spouses, siblings on board), parch (number of parents, children on board), pclass (indication of social class -> 1 is highest, 3 lowest)
# drop the rest of the features for now
data.drop(columns=["name", "ticket", "cabin", "boat", "body", "home.dest"], inplace=True)

print(data.info())

# Assuming 'data' is your DataFrame
mean_age = data['age'].mean()

# Replace missing values in 'age' with the mean
data['age'].fillna(mean_age, inplace=True)

# delete or replace null values in row -> first try: delete
data.dropna(axis=0, subset=["age", "embarked", "fare"], inplace=True)
data.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
data.replace({'embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

print(data.info())



# making a correlation matrix to see which features are most important
corr_matrix = data[["survived", "sex", "pclass", "parch", "fare", "embarked"]].corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
plt.show()

# dataTitanic visualisation
fare_bins = [0, 20, 50, 100, 200, 1000]
fare_labels = ['0-20', '20-50', '50-100', '100-200', '200+']
fareGroup = pd.cut(data['fare'], bins=fare_bins, labels=fare_labels, right=False)
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 100]
age_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', "60-70", "70+"]
ageGroup = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)

fig, axes = plt.subplots(4, 2, figsize=(12, 8))

sns.barplot(x="sex", y="survived", data=data, ax=axes[0, 0])
axes[0, 0].set_title("Survival rate by sex")
sns.barplot(x="pclass", y="survived", data=data, ax=axes[0, 1])
axes[0, 1].set_title("Survival rate by pclass")
sns.barplot(x="parch", y="survived", data=data, ax=axes[1, 0])
axes[1, 0].set_title("Survival rate by parch")
sns.barplot(x=fareGroup, y="survived", data=data, ax=axes[1, 1])
axes[1, 1].set_title("Survival rate by fare")
sns.barplot(x="embarked", y="survived", data=data, ax=axes[2, 0])
axes[2, 0].set_title("Survival rate by embarked")
sns.barplot(x="sibsp", y="survived", data=data, ax=axes[2, 1])
axes[2, 1].set_title("Survival rate by sibsp")
sns.barplot(x=ageGroup, y="survived", data=data, ax=axes[3, 0])
axes[3, 0].set_title("Survival rate by age")
fig.delaxes(axes[3,1])

plt.tight_layout()
plt.show()


features = ["sex", "age", "fare", "sibsp", "parch", "pclass", "embarked"]
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

