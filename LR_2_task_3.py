import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})

sns.pairplot(df, hue='target_name', diag_kind='kde')
plt.show()

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifiers = {
    'K-найближчих сусідів': KNeighborsClassifier(n_neighbors=5),
    'Метод опорних векторів': SVC(kernel='linear'),
    'Дерево рішень': DecisionTreeClassifier()
}

results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f'Класифікатор: {name}')
    print(classification_report(y_test, y_pred))
    print('Матриця невідповідностей:\n', confusion_matrix(y_test, y_pred))
    print('-' * 40)

plt.bar(results.keys(), results.values(), color=['blue', 'green', 'red'])
plt.xlabel('Класифікатор')
plt.ylabel('Точність')
plt.title('Порівняння точності класифікаторів')
plt.ylim(0, 1)
plt.show()
