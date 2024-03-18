#%%
import numpy as np
import matplotlib.pyplot as plt
from pde import CartesianGrid, solve_laplace_equation, PolarSymGrid
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
#initialise domain in polar coordinates
r=np.linspace(0,1,100)
theta=np.linspace(0,2*np.pi,100)
r,theta=np.meshgrid(r,theta)
#convert to X,Y for interpolation later
X=r*np.cos(theta)
Y=r*np.sin(theta)
positions = np.transpose(np.vstack([X.ravel(), Y.ravel()]))

#define grid in pde library
grid = CartesianGrid([[-2, 2]] * 2, 250)
# grid=PolarSymGrid(radius=(1,2), shape=10)

#define bcs, 
bc_x = [{"value": 10}, {"value": 11}]
bc_y = [{"value": 12}, {"value": 13}]
bcs= [bc_x,bc_y]

#solve
res = solve_laplace_equation(grid, bcs)
res.plot()

# extract data in matrix form
interpolated_data=np.reshape(res.interpolate(positions),(100,100))
# %%
fig, ax = plt.subplots(figsize=(10,10),subplot_kw={'projection': 'polar'})

ax.xaxis.grid(True)
ax.xaxis.set_ticklabels([])
contours=ax.contourf(theta,r,interpolated_data,alpha=1,zorder=1)

ax.set_rmax(1)
ax.set_rticks([0.25,0.5,0.75,1])
cbar = fig.colorbar(contours, ax=ax)
background= plt.imread("Mercator_projection_Square.jpeg")
aircraft=plt.imread("aircraft.png")
ax_image = fig.add_axes(ax.get_position())
ax_image.imshow(background, alpha=0.2,zorder=0)
ax_image.axis('off')
imagebox = OffsetImage(aircraft, zoom=0.01, alpha=None)
ab = AnnotationBbox(imagebox, (0.5, 0.5), xycoords='axes fraction',bboxprops=dict(facecolor='black', edgecolor='black'))
ax_image.add_artist(ab)
# ax.set_theta_zero_location('N')
fig.savefig("contours.svg")
# %%
