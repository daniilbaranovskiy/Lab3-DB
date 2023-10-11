import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Згенеруємо випадкові дані
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)


# Функція для побудови кривих навчання
def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5,
        scoring='neg_mean_squared_error'
    )

    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    test_rmse = np.sqrt(-test_scores.mean(axis=1))

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_rmse, label='Навчання')
    plt.plot(train_sizes, test_rmse, label='Перевірка')
    plt.xlabel('Розмір навчального набору')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.title('Криві навчання')
    plt.show()


# Побудуємо криві навчання для лінійної регресії
lin_reg = LinearRegression()
plot_learning_curve(lin_reg, X, y)

# Побудуємо криві навчання для поліноміальної регресії (2-го ступеня)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
poly_reg = LinearRegression()
plot_learning_curve(poly_reg, X_poly, y)
