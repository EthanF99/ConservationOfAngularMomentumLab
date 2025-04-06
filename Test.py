import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import numpy as np

names = {
    1: "Tail",
    2: "FrontRight",
    3: "FrontLeft",
    4: "BackLeft",
    5: "BackRight",
    6: "Center",
    7: "Head",
}

# Path to your CSV file
csv_file = r"C:\Users\ethan\PycharmProjects\ConservationOfAngularMomentumLab\Data\csvxypts.csv"

# Load the data from CSV
data = pd.read_csv(csv_file)

# Check the first few rows to understand the format
print(data.head())

# Extract columns for x and y coordinates of tracked points (e.g., pt1_cam1_X, pt1_cam1_Y, ...)
points_x = [col for col in data.columns if col.endswith('_X')]
points_y = [col for col in data.columns if col.endswith('_Y')]

# Create lists to hold the x, y coordinates for plotting
x_coords = [data[pt_x].values for pt_x in points_x]
y_coords = [data[pt_y].values for pt_y in points_y]

# Time is generally the index in the CSV or a separate column
time = np.arange(len(data))  # If no explicit time column, use the index as time

# Plot the X coordinates
plt.figure(figsize=(12, 8))
for i, x in enumerate(x_coords):
    # Get point number from column name (assuming format pt1_cam1_X, pt2_cam1_X, etc.)
    point_num = i + 1
    point_name = names.get(point_num, f"Point {point_num}")
    plt.plot(time, x, label=f'{point_name} X', linestyle='--')

plt.xlabel("Time (frames)")
plt.ylabel("X Coordinates (pixels)")
plt.title("X Coordinates of Tracked Points Over Time")
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('tracked_x_coordinates.png')

# Plot the Y coordinates
plt.figure(figsize=(12, 8))
for i, y in enumerate(y_coords):
    point_num = i + 1
    point_name = names.get(point_num, f"Point {point_num}")
    plt.plot(time, y, label=f'{point_name} Y')

plt.xlabel("Time (frames)")
plt.ylabel("Y Coordinates (pixels)")
plt.title("Y Coordinates of Tracked Points Over Time")
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('tracked_y_coordinates.png')

# Get center of mass coordinates (point 6)
com_index = 5  # 0-based index for point 6
x_com = x_coords[com_index]
y_com = y_coords[com_index]

# Calculate velocity for each point relative to COM
vx_rel = []
vy_rel = []

for i in range(len(x_coords)):
    # Position relative to COM
    rx = x_coords[i] - x_com
    ry = y_coords[i] - y_com

    # Calculate velocity of this point relative to COM
    vx_rel.append(np.gradient(rx))
    vy_rel.append(np.gradient(ry))

# Calculate angular momentum around COM for each point (excluding the COM itself)
angular_momentum = np.zeros(len(data))

for i in range(len(x_coords)):
    if i == com_index:  # Skip the COM point
        continue

    # Position relative to COM
    rx = x_coords[i] - x_com
    ry = y_coords[i] - y_com

    # Angular momentum contribution using relative velocities
    # L = r × v_rel = rx*vy_rel - ry*vx_rel
    angular_momentum += rx * vy_rel[i] - ry * vx_rel[i] #Using the cross product of r and velocity

mean_angular_momentum = float(np.nanmean(angular_momentum))

# Plot the angular momentum over time
plt.figure(figsize=(10, 6))
plt.plot(angular_momentum, linewidth=2)
plt.axhline(y=mean_angular_momentum, color='r', linestyle='--', alpha=0.7,
           label=f'Mean: {mean_angular_momentum:.2f}')
plt.xlabel('Frame')
plt.ylabel('Angular Momentum (arbitrary units)')
plt.title('Angular Momentum During Cat Falling (Relative to COM)')
plt.grid(True)
plt.legend()
plt.savefig('angular_momentum_relative.png')


# Calculate and plot net rotation
# Use the angle between two vectors (e.g., head-COM and tail-COM)
head_index = 6  # 0-based index for point 7 (head)
tail_index = 0  # 0-based index for point 1 (tail)

angles = []
for i in range(len(data)):
    # Vector from COM to head
    head_vec = [x_coords[head_index][i] - x_com[i],
                y_coords[head_index][i] - y_com[i]]

    # Vector from COM to tail
    tail_vec = [x_coords[tail_index][i] - x_com[i],
                y_coords[tail_index][i] - y_com[i]]

    # Calculate angle between vectors
    dot_product = head_vec[0] * tail_vec[0] + head_vec[1] * tail_vec[1]
    head_mag = np.sqrt(head_vec[0] ** 2 + head_vec[1] ** 2)
    tail_mag = np.sqrt(tail_vec[0] ** 2 + tail_vec[1] ** 2)

    cos_angle = dot_product / (head_mag * tail_mag)
    # Ensure within valid range due to potential numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    angles.append(np.degrees(angle))

plt.figure(figsize=(10, 6))
plt.plot(angles, linewidth=2)
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.title('Head-COM-Tail Angle During Cat Falling')
plt.grid(True)
plt.savefig('cat_configuration_angle.png')



# Convert angle list to a numpy array in radians
angles_rad = np.radians(angles)

# Angular velocity = derivative of angle (in radians) with respect to time
omega_rad = np.gradient(angles_rad)  # If time step is 1 frame per unit



masses = np.ones(len(x_coords))  # Assuming equal mass
#masses = np.array([0.12, 0.20, 0.20, 0.24, 0.24, 2.40, 0.60]) #average masses of different parts of a cat

I = np.zeros(len(data))

for i in range(len(x_coords)):
    if i == com_index:
        continue
    rx = x_coords[i] - x_com
    ry = y_coords[i] - y_com
    r_squared = rx**2 + ry**2
    I += masses[i] * r_squared  # Or just I += r_squared if mass=1



#x_head0 = x_coords[6][0]
#y_head0 = y_coords[6][0]
#x_com0 = x_coords[5][0]
#y_com0 = y_coords[5][0]

# Compute pixel distance from head to center
#head_to_com_px = np.sqrt((x_head0 - x_com0)**2 + (y_head0 - y_com0)**2)
#body_length_px = 2 * head_to_com_px

# Estimate torso length (about 40% of body) and radius (~15% of that)
#torso_length_px = 0.4 * body_length_px
#torso_radius_px = 0.15 * torso_length_px


#Inertia of Cylinder I = 0.5mr^2
#average torso weight of a housecat = 6 kg
#I += 0.5*6*torso_radius_px**2 #


# Angular momentum
L = I * omega_rad

# Plot angular momentum
plt.figure(figsize=(10, 6))
plt.plot(L, linewidth=2)
plt.axhline(y=float(np.nanmean(L)), color='r', linestyle='--', alpha=0.7,
           label=f'Mean: {float(np.nanmean(L)):.2f}')
plt.xlabel('Frame')
plt.ylabel('Angular Momentum (Iω, arbitrary units)')
plt.title('Angular Momentum Through L=Iω')
plt.grid(True)
plt.legend()
plt.savefig('angular_momentum_lw.png')


# Plot the moment of inertia over time
plt.figure(figsize=(10, 6))
plt.plot(I, label="Moment of Inertia")
plt.xlabel('Frame')
plt.ylabel('Moment of Inertia (arbitrary units)')
plt.title('Moment of Inertia Over Time')
plt.grid(True)
plt.legend()
plt.savefig('moment_of_inertia.png')

