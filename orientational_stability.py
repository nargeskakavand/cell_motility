# Filename: particle_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit

# Import required libraries
import os

# Define conversion factors
time_step = 1 / 16
scaling_factor = 1000 / 359

# Load data from CSV file
data = pd.read_csv('GCTot_resultF.csv')
particles = data.particle.unique()
particle_list = particles.tolist()
particle_frames = {p: pd.DataFrame for p in particles}  # Empty dictionary for individual dataframes

# Generate dataframes for each particle
for p in particle_frames.keys():
    particle_frames[p] = data[data.particle == p]

print(f"Number of unique particles: {len(particle_list)}")

# Define helper functions
def compute_angles(x_vals, y_vals, time_vals):
    x_diff, y_diff, delta_time, theta_list, theta_values = [], [], [], [], []
    for i in range(len(x_vals) - 1):
        x_diff.append(x_vals[i + 1] - x_vals[i])
        y_diff.append(y_vals[i + 1] - y_vals[i])
        delta_time.append((time_vals[i + 1] - time_vals[i]) * time_step)

    for j in range(len(x_diff)):
        x_velocity = x_diff[j]
        y_velocity = y_diff[j]

        if x_velocity == 0 or y_velocity == 0:
            print('Warning: Possible error at index', j)
        if x_velocity > 0 and y_velocity < 0:
            angle = np.pi + np.arctan(x_velocity / y_velocity)
        elif x_velocity < 0 and y_velocity < 0:
            angle = -np.pi + np.arctan(x_velocity / y_velocity)
        else:
            angle = np.arctan(x_velocity / y_velocity)

        if angle > np.pi or angle < -np.pi:
            print('Angle out of bounds:', x_velocity, y_velocity, angle)

        theta_values.append((angle, j))

    theta_values = sorted(theta_values, key=lambda x: x[1])
    theta_sorted, _ = list(zip(*theta_values))
    theta_list = list(theta_sorted)

    return theta_list, x_diff, y_diff, delta_time

def angular_velocity(theta, delta_t):
    angle_changes = []
    next_theta = []

    for i in range(len(theta) - 1):
        start_theta = theta[i]
        end_theta = theta[i + 1]
        if 0 < start_theta < np.pi and 0 < end_theta < np.pi:
            delta = end_theta - start_theta
        elif -np.pi < start_theta < 0 and -np.pi < end_theta < 0:
            delta = end_theta - start_theta
        elif 0 < start_theta < np.pi and -np.pi < end_theta < 0:
            delta = (2 * np.pi) - (end_theta + start_theta)
        elif -np.pi < start_theta < 0 and 0 < end_theta < np.pi:
            delta = -((2 * np.pi) - (end_theta + start_theta))
        else:
            delta = 0  # Handle unexpected cases
        angle_changes.append((delta, i))
        next_theta.append((end_theta, i))

    angle_changes = sorted(angle_changes, key=lambda x: x[1])
    delta_theta, _ = list(zip(*angle_changes))
    next_theta = sorted(next_theta, key=lambda x: x[1])
    theta_final, _ = list(zip(*next_theta))

    delta_theta_list = list(delta_theta)
    final_theta = list(theta_final)
    delta_time_adj = delta_t[1:]

    angular_speed = list(map(lambda x, y: x / y, delta_theta_list, delta_time_adj))

    return final_theta, angular_speed

def categorize_by_theta(omega, theta, bin_size):
    theta_degrees = np.degrees(theta)
    categorized_omega = []
    categorized_theta = []
    bins = np.arange(-180, 180, bin_size)

    for i in range(len(bins) - 1):
        omega_values = [omega[j] for j in range(len(omega)) if bins[i] < theta_degrees[j] <= bins[i + 1]]

        if omega_values:
            avg_omega = np.mean(omega_values)
            categorized_theta.append((bins[i] + bins[i + 1]) / 2)
            categorized_omega.append(avg_omega)

    # Handle outliers
    for idx in range(len(categorized_theta)):
        if categorized_omega[idx] > 0.8 or categorized_omega[idx] < -0.8:
            categorized_omega[idx] = 0

    return categorized_theta, categorized_omega

def fit_sine_curve(omega_data, theta_data, initial_amplitude):
    # Fit data to a sinusoidal model and save the plot as a PDF
    theta_radians = np.radians(np.linspace(min(theta_data), max(theta_data), len(theta_data)))

    initial_frequency = 1 / 180
    initial_amplitude = 3 * np.std(omega_data) / (2 ** 0.5)
    initial_phase = 0
    initial_offset = np.mean(omega_data)

    initial_params = [initial_frequency, initial_amplitude, initial_phase, initial_offset]

    # Sine function for fitting
    def sine_model(x, freq, amp, phase, offset):
        return np.sin(x * freq + phase) * amp + offset

    # Perform curve fitting
    fit_parameters, _ = curve_fit(sine_model, theta_radians, omega_data, p0=initial_params)
    fitted_curve = sine_model(theta_radians, *fit_parameters)

    # Extract and adjust offset for centering
    offset = fit_parameters[3]
    centered_data = omega_data - offset
    centered_fitted_curve = fitted_curve - offset

    # Set symmetric y-axis limits
    max_value = max(abs(centered_data).max(), abs(centered_fitted_curve).max())
    y_axis_limits = (-max_value, max_value)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.errorbar(np.degrees(theta_radians), centered_data, fmt='o', label='Observed Data', color='k', alpha=0.5)
    plt.plot(np.degrees(theta_radians), centered_fitted_curve, '^', color='r', label='Fitted Sine Curve')

    plt.ylabel('Angular Speed (rad/sec)', fontsize=16, fontweight='bold')
    plt.xlabel('Angle (degrees)', fontsize=16, fontweight='bold')
    plt.ylim(y_axis_limits)
    plt.legend()
    plt.title('Angular Speed vs Angle with Sine Curve Fit')

    # Save plot as PDF
    plt.savefig('angular_speed_angle_fit.pdf', format='pdf')
    plt.close()

    print('***************')
    print('Amplitude Range:', min(centered_fitted_curve), max(centered_fitted_curve))
    print('Amplitude (Peak-to-Peak):', max(centered_fitted_curve) - min(centered_fitted_curve))
    print('Plot saved as angular_speed_angle_fit.pdf')
    print('***************')

# Script execution
if __name__ == '__main__':
    scaling_values = []
    mean_speeds = []
    x_final = np.array([])
    y_final = np.array([])

    angle_list = []
    angular_speeds = []

    for particle in particle_list:
        data_segment = particle_frames[particle]
        if len(data_segment) > 100:  # Modify as needed
            x_values = data_segment['x'].values
            y_values = data_segment['y'].values * -1
            time_values = data_segment['frame'].values

            dx = x_values[-1] - x_values[0]
            dy = y_values[-1] - y_values[0]
            scale = np.sqrt(dx ** 2 + dy ** 2) * time_step * scaling_factor
            scaling_values.append(scale * (1 / time_step) * (1 / scaling_factor))

            if 3 < scale < 25:
                avg_y_gradient = np.mean(np.gradient(y_values))
                smooth_interp, param_values = interpolate.splprep([x_values, y_values], k=2, s=1e6, per=0,
                                                                  u=np.linspace(min(x_values), max(x_values), len(x_values)))
                x_interp, y_interp = interpolate.splev(np.linspace(min(x_values), max(x_values), num=len(x_values)), smooth_interp)

                angles, x_diff, y_diff, time_deltas = compute_angles(x_interp, y_interp, time_values)
                angle_list.extend(angles)
                final_angles, angular_change = angular_velocity(angles, time_deltas)
                angular_speeds.extend(angular_change)
                mean_speeds.append(np.mean(np.sqrt(np.array(y_diff) ** 2) / np.array(time_deltas)) * (1 / time_step))

                x_final = np.concatenate((x_final, x_interp))
                y_final = np.concatenate((y_final, y_interp))

    theta_bins, omega_binned = categorize_by_theta(angular_speeds, angle_list, 50)  # Adjust bin size if necessary

    fit_sine_curve(omega_binned, theta_bins, 0.2)
