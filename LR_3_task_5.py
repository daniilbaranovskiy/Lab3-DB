import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Згенеруємо випадкові дані за першим варіантом
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Побудуємо модель лінійної регресії
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Побудуємо модель поліноміальної регресії (2-го ступеня)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Виведемо коефіцієнти лінійної регресії
print("Коефіцієнти лінійної регресії:")
print("Коефіцієнт (вага) X:", lin_reg.coef_)
print("Перетин лінії:", lin_reg.intercept_)

# Виведемо коефіцієнти поліноміальної регресії
print("\nКоефіцієнти поліноміальної регресії:")
print("Коефіцієнти X^2, X та константа:", poly_reg.coef_)
print("Перетин лінії:", poly_reg.intercept_)

# Побудуємо графік
X_new = np.linspace(-5, 5, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)

y_lin = lin_reg.predict(X_new)
y_poly = poly_reg.predict(X_new_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Дані", color='b')
plt.plot(X_new, y_lin, label="Лінійна регресія", color='g', linewidth=2)
plt.plot(X_new, y_poly, label="Поліноміальна регресія", color='r', linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Модель регресії")
plt.grid(True)
plt.show()
