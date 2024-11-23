import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])

z = np.polyfit(x, y, 3)
p = np.poly1d(z)
x_new = np.linspace(x[0], x[-1], 100)
y_new = p(x_new)

plt.scatter(x, y, color='red', label='Data Points')
plt.plot(x_new, y_new, color='blue', label=f'3rd-degree Polynomial Fit')
plt.legend()
plt.show()