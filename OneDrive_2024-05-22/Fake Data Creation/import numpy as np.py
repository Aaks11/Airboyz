import numpy as np
import matplotlib.pyplot as plt

# Define grid size
grid_size = 10

# Generate random velocities for each grid point
velocities_x = np.random.uniform(-1, 1, size=(grid_size, grid_size))
velocities_y = np.random.uniform(-1, 1, size=(grid_size, grid_size))

# Generate grid coordinates
x = np.arange(0, grid_size, 1)
y = np.arange(0, grid_size, 1)

# Create meshgrid
X, Y = np.meshgrid(x, y)

# Plot streamlines
plt.streamplot(X, Y, velocities_x, velocities_y, density=2, arrowsize=1)
plt.xlim(0, grid_size - 1)
plt.ylim(0, grid_size - 1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Streamlines on Cartesian Grid')
plt.grid(True)
plt.show()
