import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

data = pd.read_csv('income_data.txt')

for column in data.select_dtypes(include=['object']).columns:
    data[column] = LabelEncoder().fit_transform(data[column])

y = data.iloc[:, -1]
X = data.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    'Логістична регресія': LogisticRegression(),
    'Лінійний дискримінантний аналіз': LinearDiscriminantAnalysis(),
    'Метод k-найближчих сусідів': KNeighborsClassifier(),
    'Дерево ухвалення рішень': DecisionTreeClassifier(),
    'Наївний Байєс': GaussianNB(),
    'Метод опорних векторів': SVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        'Точність': accuracy_score(y_test, y_pred),
        'Прецизійність': precision_score(y_test, y_pred, average='weighted'),
        'Повнота': recall_score(y_test, y_pred, average='weighted'),
        'F1-міра': f1_score(y_test, y_pred, average='weighted'),
        'Звіт класифікації': classification_report(y_test, y_pred)
    }

for name, metrics in results.items():
    print(f'=== {name} ===')
    for metric, value in metrics.items():
        if metric != 'Звіт класифікації':
            print(f'{metric}: {value:.4f}')
        else:
            print(f'{metric}:{value}')
            print('\n' + '-' * 40 + '\n')
