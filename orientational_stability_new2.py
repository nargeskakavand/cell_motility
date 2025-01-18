import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

##############################################################################
# 1) Utility Functions
##############################################################################

def wrap_angle_180(angle_deg):
    """
    Wrap an angle in degrees into the range [-180, 180).
    Example: +190 -> -170, -540 -> 180, etc.
    """
    return ((angle_deg + 180) % 360) - 180

def sine_func(theta_deg, A, phi, gamma=0):
    """
    Enhanced model function for fitting: omega = A * sin(theta + phi) + gamma
    where 'theta_deg' is in degrees.
    Parameters:
        - A: Amplitude
        - phi: Phase shift (in radians)
        - gamma: Vertical offset
    """
    theta_rad = np.deg2rad(theta_deg)
    return A * np.sin(theta_rad + phi) + gamma

##############################################################################
# 2) Main
##############################################################################

def main():
    ##########################################################################
    # A) LOAD DATA
    ##########################################################################
    # Update with your actual CSV filename/path
    experiment_round = 1
    sample_type = 'HA452'
    day = 1
    sample_num = 1
    base_path = f"F:/perturbation_all_rounds/reorientation_new_data/{experiment_round}/" \
                f"{sample_type}/{day}/{sample_num}"
    filename = "T1tracked_particles.csv"
    df_all = pd.read_csv(os.path.join(base_path, filename))
    df_all = pd.read_csv("T1tracked_particles.csv")

    MIN_FRAMES = 20

    # 1) Filter out groups with fewer than MIN_FRAMES
    df_filtered = df_all.groupby("particle").filter(lambda g: len(g) >= MIN_FRAMES)

    # 2) Now group the filtered DataFrame
    grouped = df_filtered.groupby("particle")

    # Prepare to collect all (theta, omega) pairs from all particles
    theta_list = []
    omega_list = []

    # Frames per second
    FPS = 16.0

    ##########################################################################
    # B) PROCESS EACH PARTICLE
    ##########################################################################
    for particle_id, df_part in grouped:
        # Sort frames in ascending order
        df_part_sorted = df_part.sort_values(by='frame')

        # Extract as NumPy arrays
        x_vals = df_part_sorted['x'].to_numpy()
        y_vals = df_part_sorted['y'].to_numpy()
        frame_vals = df_part_sorted['frame'].to_numpy()

        # If fewer than 2 frames, skip (can't compute velocity)
        if len(x_vals) < 2:
            continue

        # 1. dx, dy for frames 1..N-1
        dx = x_vals[1:] - x_vals[:-1]
        dy = y_vals[1:] - y_vals[:-1]

        # 2. Orientation angle in radians
        theta_rad = np.arctan2(dy, dx)  # length = N-1

        # 3. Unwrap to avoid ±180 jumps
        theta_unwrapped = np.unwrap(theta_rad)  # length = N-1

        # 4. Convert to degrees
        theta_deg_array = np.degrees(theta_unwrapped)  # length = N-1

        # 5. Time difference
        dframe = frame_vals[1:] - frame_vals[:-1]  # length = N-1
        dt_array = dframe / FPS                    # length = N-1

        # 6. Angular velocity: dtheta / dt
        if len(theta_unwrapped) < 2:
            continue
        dtheta_rad = theta_unwrapped[1:] - theta_unwrapped[:-1]  # length = N-2
        dt_for_omega = dt_array[1:]                              # length = N-2
        omega_rad_s_array = dtheta_rad / dt_for_omega            # length = N-2

        # We'll associate omega[i] with the angle at index i+1
        # So let's skip the first angle in theta_deg_array
        # to align them in length (N-2 for both).
        if len(omega_rad_s_array) < 1:
            continue

        theta_for_omega = theta_deg_array[1:]    # length = N-2
        # Now we have matched arrays: (theta_for_omega, omega_rad_s_array)

        theta_list.append(theta_for_omega)
        omega_list.append(omega_rad_s_array)

    ##########################################################################
    # C) COMBINE ALL PARTICLES
    ##########################################################################
    if len(theta_list) == 0 or len(omega_list) == 0:
        print("No valid data across all particles!")
        return

    theta_all = np.concatenate(theta_list)
    omega_all = np.concatenate(omega_list)

    ##########################################################################
    # D) FILTER OUTLIERS (OPTIONAL)
    ##########################################################################
    mask = (omega_all < 1.0) & (omega_all > -1.0)
    theta_filt = theta_all[mask]
    omega_filt = omega_all[mask]

    print("theta_filt length =", len(theta_filt))
    print("omega_filt length =", len(omega_filt))

    if len(theta_filt) == 0:
        print("Error: No data available for curve fitting!")
        return

    ##########################################################################
    # E) WRAP ANGLES TO [-180, 180)
    ##########################################################################
    theta_wrapped = np.array([wrap_angle_180(a) for a in theta_filt])

    ##########################################################################
    # F) BIN THE DATA (e.g., 18° increments)
    ##########################################################################
    bin_edges = np.arange(-180, 181, 18)  # -180, -162, -144, ... +180
    bin_index = np.digitize(theta_wrapped, bin_edges)

    df_binning = pd.DataFrame({
        'theta_deg': theta_wrapped,
        'omega': omega_filt,
        'bin': bin_index
    })

    # Group by bin and compute mean ± std for each bin
    bin_stats = df_binning.groupby('bin').agg(
        theta_mean=('theta_deg', 'mean'),
        omega_mean=('omega', 'mean'),
        omega_std=('omega', 'std'),
        count=('omega', 'count')
    ).reset_index()

    # Drop bins with no data
    bin_stats.dropna(subset=['omega_mean'], inplace=True)

    ##########################################################################
    # G) FIT SINE CURVE TO BINS
    ##########################################################################
    theta_binned = bin_stats['theta_mean'].values
    omega_binned = bin_stats['omega_mean'].values

    if len(theta_binned) < 3:
        print("Not enough binned data for curve fitting.")
        return

    # Initial guesses for A, phi
    A_initial = (np.max(omega_binned) - np.min(omega_binned)) / 2
    phi_initial = 0.0  # Start with no phase shift
    initial_guesses = [A_initial, phi_initial]

    try:
        popt, pcov = curve_fit(sine_func, theta_binned, omega_binned, p0=initial_guesses)
        A_fitted, phi_fitted = popt
        beta_binned = popt[0]
        tau_r_binned = 1.0 / (2.0 * beta_binned)

        print(f"Sine fit (binned) -> beta = {beta_binned:.6f}, tau_r = {tau_r_binned:.6f} s")
        print(f"Sine fit -> A = {A_fitted:.6f}, phi = {phi_fitted:.6f} rad")
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        return

    ##########################################################################
    # H) PLOT
    ##########################################################################
    plt.figure(figsize=(9, 6))

    # 1. Plot the binned data without error bars
    plt.scatter(
        bin_stats['theta_mean'],
        bin_stats['omega_mean'],
        color='limegreen',
        label='Binned Data (18° increments)',
        zorder=6
    )

    # 2. Plot the fitted sine curve
    # theta_plot = np.linspace(-180, 180, 300)
    # omega_plot = sine_func(theta_plot, A_fitted, phi_fitted, gamma_fitted)
    # plt.plot(theta_plot, omega_plot, 'r--', lw=2, label='Fitted Sine')

    omega_fit = sine_func(bin_stats['theta_mean'], A_fitted, phi_fitted)
    plt.scatter(
        bin_stats['theta_mean'],
        omega_fit,
        color='orangered',
        label='Fitted Sine',
        zorder=5
    )

    plt.xlabel("Orientation angle (degrees)", fontsize=12)
    plt.ylabel("Angular velocity (rad/s)", fontsize=12)
    plt.title("Binned & Fitted Orientation vs. Angular Velocity", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
