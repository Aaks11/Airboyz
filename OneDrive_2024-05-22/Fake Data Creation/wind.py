#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# Define grid size
n=1000
base_vcity=40
base_direction=120
bds_velocity=np.array([-20, 20])
bds_direction=np.array([-30,30])

velocities=base_vcity+np.random.uniform(bds_velocity[0],bds_velocity[1],size=(n,n))
directions=base_direction+np.random.uniform(bds_direction[0],bds_direction[1],size=(n,n))
directions=(360-directions+90)*np.pi/180

velocities_x=velocities*np.cos(directions)
velocities_y=velocities*np.sin(directions)

#%%
# Generate random velocities for each grid point

# velocities_x = np.random.uniform(-1, 1, size=(grid_size, grid_size))


# Generate grid coordinates
x = np.linspace(-1,1,n)
y = np.linspace(-1,1,n)

# Create meshgrid
X, Y = np.meshgrid(x, y)

# Plot streamlines
lw=2*velocities/np.max(velocities)
plt.figure(figsize=(10, 8))
plt.streamplot(X, Y, velocities_x, velocities_y, color=velocities,density=0.75, cmap='autumn', linewidth=lw, arrowstyle='-|>', integration_direction='forward',zorder=0,broken_streamlines=False)
# Calculate the distance from the origin
distance = np.sqrt(X**2 + Y**2)
colors = [(1, 1, 1, 0), (1, 1, 1, 1)]  # Transparent, White
cmap_name = 'outside_only'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)
# Define the condition to check if the point is outside the circle
outside_circle = distance > 1
plt.pcolormesh(X, Y, outside_circle, cmap=cm, shading='auto',zorder=1)
circle = plt.Circle((0, 0), 1, color='black', fill=False)
plt.gca().add_patch(circle)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.grid(True)
plt.axis("off")
plt.gca().set_aspect('equal')
plt.savefig("wind.png",dpi=600)
plt.show()

#%%

