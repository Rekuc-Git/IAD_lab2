import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

X, y = datasets.make_moons(n_samples=300, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def train_and_evaluate_svm(kernel, **kwargs):
    model = SVC(kernel=kernel, **kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"SVM з {kernel} ядром:")
    print(classification_report(y_test, y_pred))
    print(f"Точність: {accuracy_score(y_test, y_pred):.4f}\n")

    plt.figure(figsize=(6, 5))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(f"SVM з {kernel} ядром")
    plt.show()


train_and_evaluate_svm('poly', degree=8)

train_and_evaluate_svm('rbf')

train_and_evaluate_svm('sigmoid')
