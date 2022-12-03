import os
import numpy as np
import fdtd
import fdtd.backend as bd
import matplotlib.pyplot as plt

#fdtd.set_backend("numpy")
fdtd.set_backend("torch")

# GRID
grid = fdtd.Grid(shape=(260, 260, 1e-4), grid_spacing=77.5e-9)

# x boundaries
grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")

# y boundaries
grid[:, 0:10, :] = fdtd.PML(name="pml_ylow")
grid[:, -10:, :] = fdtd.PML(name="pml_yhigh")


# z boundaries
grid[:, :, 0:10] = fdtd.PML(name="pml_zlow")
grid[:, :, -10:] = fdtd.PML(name="pml_zhigh")

simfolder = grid.save_simulation("Lenses")  # initializing environment to save simulation data
print(simfolder)

# OBJECTS
#x, y, z = np.arange(-200, 200, 1), np.arange(-200, 200, 1), np.arange(190, 200, 1)
#X, Y, Z = np.meshgrid(x, y, z)
#lens_mask = X ** 2 + Y ** 2 + Z ** 2 <= 40000
#for j, col in enumerate(lens_mask.T):
#    for i, val in enumerate(np.flip(col)):
#        if val:
#            grid[30 + i : 50 - i, 30+i : 50-i, j-100:j-99] = fdtd.Object(permittivity=1.5 ** 2, name=str(i) + "," + str(j))
#            break

# OBJECTS
refractive_index = 1.7
x = y = z= np.linspace(-1,1,100)
X, Y, Z = np.meshgrid(x, y, z)
circle_mask = X**2 + Y**2 + Z**2 < 1
permittivity = np.ones((100,100,100))
permittivity += circle_mask[:,:,:]*(refractive_index**2 - 1)
grid[170:270, 170:270, 400:500] = fdtd.Object(permittivity=permittivity, name="object")

# SOURCE
#grid[15, 50:150, 0] = fdtd.LineSource(period=1550e-9 / (3e8), name="source")
grid[:, :, 20] = fdtd.PlaneSource(period=1150e-9 / (3e8), name="source")

# DETECTORS
grid[80:200, 80:200, -20] = fdtd.BlockDetector(name="detector")

# SAVE GEOMETRY
with open(os.path.join(simfolder, "grid.txt"), "w") as f:
    f.write(str(grid))
    wavelength = 3e8/grid.source.frequency
    wavelengthUnits = wavelength/grid.grid_spacing
    GD = np.array([grid.x, grid.y, grid.z])
    gridRange = [np.arange(x/grid.grid_spacing) for x in GD]
    objectRange = np.array([[gridRange[0][x.x], gridRange[1][x.y], gridRange[2][x.z]] for x in grid.objects], dtype=object).T
    f.write("\n\nGrid details (in wavelength scale):")
    f.write("\n\tGrid dimensions: ")
    f.write(str(GD/wavelength))
    f.write("\n\tSource dimensions: ")
    f.write(str(np.array([grid.source.x[-1] - grid.source.x[0] + 1, grid.source.y[-1] - grid.source.y[0] + 1, grid.source.z[-1] - grid.source.z[0] + 1])/wavelengthUnits))
    f.write("\n\tObject dimensions: ")
    f.write(str([(max(map(max, x)) - min(map(min, x)) + 1)/wavelengthUnits for x in objectRange]))

# SIMULATION
from IPython.display import clear_output # only necessary in jupyter notebooks
for i in range(400):
    grid.step()  # running simulation 1 timestep a time and animating
    if i % 10 == 0:
        # saving frames during visualization
        grid.visualize(x=0, animate=True, index=i, save=True, folder=simfolder)
        plt.title(f"{i:3.0f}")
        clear_output(wait=True) # only necessary in jupyter notebooks

grid.save_data()  # saving detector readings

try:
    video_path = grid.generate_video(delete_frames=False)  # rendering video from saved frames
except:
    video_path = ""
    print("ffmpeg not installed?")

if video_path:
    from IPython.display import Video
    #display(Video(video_path, embed=True))
    Video(video_path, embed=True)

df = np.load(os.path.join(simfolder, "detector_readings.npz"))
fdtd.dB_map_2D(df["detector (E)"])
