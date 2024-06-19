import numpy as np
import matplotlib.pyplot as plt

# Генеруємо дані для прикладу
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)

# Додаємо одиничний стовпець для методу найменших квадратів
X_b = np.c_[np.ones((100, 1)), X]

# Градієнтний спуск
def gradient_descent(X, Y, learning_rate=0.1, n_iter=1000):
    m = len(Y)
    theta = np.random.randn(2, 1)
    loss_history = []
    
    for iteration in range(n_iter):
        gradients = 2/m * X.T @ (X @ theta - Y)
        theta = theta - learning_rate * gradients
        loss = np.mean((X @ theta - Y) ** 2)
        loss_history.append(loss)
        
    return theta, loss_history

theta_gd, loss_history = gradient_descent(X_b, Y)
print(f"Градієнтний спуск: intercept={theta_gd[0][0]}, slope={theta_gd[1][0]}")

# Побудова графіка регресії
plt.figure(figsize=(10, 6))
plt.plot(X, Y, "b.")
plt.plot(X, X_b @ theta_gd, "g-", label="Gradient Descent")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Лінійна регресія: метод найменших квадратів та градієнтний спуск")
plt.legend()
plt.show()

# Графік похибки від кількості ітерацій
plt.figure(figsize=(10, 6))
plt.plot(range(len(loss_history)), loss_history, "b-")
plt.xlabel("Кількість ітерацій")
plt.ylabel("MSE")
plt.title("Графік похибки від кількості ітерацій")
plt.show()
