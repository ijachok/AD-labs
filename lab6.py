import numpy as np
import matplotlib.pyplot as plt

# Генерація даних
np.random.seed(0)
true_k = 2.5
true_b = 1.0
x = np.random.rand(100) * 10
noise = np.random.randn(100)
y = true_k * x + true_b + noise

# Метод найменших квадратів
def least_squares(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    b1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    b0 = y_mean - b1 * x_mean
    return b0, b1

b0, b1 = least_squares(x, y)

# Оцінка за допомогою np.polyfit
polyfit_params = np.polyfit(x, y, 1)

# Порівняння знайдених параметрів
print(f"Метод найменших квадратів: b0 = {b0}, b1 = {b1}")
print(f"np.polyfit: b0 = {polyfit_params[1]}, b1 = {polyfit_params[0]}")
print(f"Початкові параметри: b0 = {true_b}, b1 = {true_k}")

# Відображення результатів
plt.scatter(x, y, label='Дані')
plt.plot(x, b0 + b1 * x, color='r', label='Регресія (метод найменших квадратів)')
plt.plot(x, polyfit_params[1] + polyfit_params[0] * x, color='g', linestyle='--', label='Регресія (np.polyfit)')
plt.plot(x, true_b + true_k * x, color='b', linestyle=':', label='Початкова пряма')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Лінійна регресія')
plt.show()



