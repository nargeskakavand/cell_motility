# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import random
import trackpy as tp
from collections import Counter
from multiprocessing import freeze_support, set_start_method


def binarize_image(image_list):
    processed_images = []
    for idx, frame in enumerate(image_list):
        # Apply adaptive thresholding and median blur
        binary_img = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
        binary_img = cv2.medianBlur(binary_img, 7)
        binary_img = binary_img[y_start:y_end, :]
        processed_images.append(binary_img)

    # Display sample binary image and save it
    fig, ax = plt.subplots()
    ax.imshow(processed_images[0], cmap='gray')
    fig.savefig(output_path + 'binary_image_sample.png')
    return processed_images


def track_particles(image_frames):
    # Perform particle tracking with batch processing and predictive linking
    particles_df = tp.batch(image_frames[:], 21, invert=True, minmass=1000, separation=7, threshold=1)
    predictor = tp.predict.NearestVelocityPredict()
    frame_groups = (frame for _, frame in particles_df.groupby('frame'))

    # Link particles across frames
    trajectories = pd.concat(
        predictor.link_df_iter(frame_groups, search_range=12, memory=5, adaptive_stop=5, adaptive_step=0.95))
    trajectories.to_csv(output_path + 'tracked_particles.csv')

    # Filter trajectories based on frame count
    particle_counts = Counter(trajectories['particle'])
    filtered_particles = [pid for pid, count in particle_counts.items() if count > min_frames]
    filtered_df = pd.DataFrame(filtered_particles, columns=['particle']).merge(trajectories, on='particle', how='inner')
    filtered_df.to_csv(output_path + 'filtered_trajectories.csv')


# Define offset for trajectory adjustments
x_offset, y_offset = 0, 30


def generate_trajectory_video(image_frames, particle_data, video_output):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = image_frames[0].shape
    video_writer = cv2.VideoWriter(video_output, fourcc, 20.0, (width, height))

    unique_particles = particle_data['particle'].unique()
    color_map = {pid: tuple(random.randint(0, 255) for _ in range(3)) for pid in unique_particles}

    for frame_num in range(len(image_frames)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for pid in unique_particles:
            particle_path = particle_data[particle_data['particle'] == pid]
            if frame_num in particle_path['frame'].values:
                positions = particle_path[particle_path['frame'] <= frame_num]
                color = color_map[pid]

                for i in range(len(positions) - 1):
                    start_pos = (int(positions.iloc[i]['x'] + x_offset), int(positions.iloc[i]['y'] + y_offset))
                    end_pos = (int(positions.iloc[i + 1]['x'] + x_offset), int(positions.iloc[i + 1]['y'] + y_offset))
                    cv2.line(frame, start_pos, end_pos, color, 2)

                current_pos = positions[positions['frame'] == frame_num]
                if not current_pos.empty:
                    x_pos, y_pos = int(current_pos.iloc[0]['x'] + x_offset), int(current_pos.iloc[0]['y'] + y_offset)
                    cv2.circle(frame, (x_pos, y_pos), radius=10, color=color, thickness=-1)

        video_writer.write(frame)

    video_writer.release()


if __name__ == '__main__':
    freeze_support()
    set_start_method("spawn")

    for experiment_round in [1]:
        for day in [2]:
            for sample_type in ['HA452']:
                for sample_num in [1]:
                    for tech_num in [1]:
                        # Define paths for input and output
                        base_path = f"round_{experiment_round}/{day}/{sample_type}/B{sample_num}/T{tech_num}/GC"
                        output_path = base_path
                        print(
                            f"Processing folder: round_{experiment_round} {day}_{sample_type}_B{sample_num}_T{tech_num}")

                        min_frames = 45
                        y_start, y_end = 30, 1850
                        frames_to_analyze = (0, 160)

                        if not os.path.exists(output_path):
                            os.makedirs(output_path)

                        # Load images
                        image_files = sorted(os.listdir(base_path))[frames_to_analyze[0]:frames_to_analyze[1]]
                        image_stack = [cv2.imread(os.path.join(base_path, filename), 0) for filename in image_files]
                        frame_height, frame_width = image_stack[0].shape

                        # Process images and perform tracking
                        binary_images = binarize_image(image_stack)
                        track_particles(binary_images)

                        # Create trajectory video
                        tracked_data = pd.read_csv(output_path + 'tracked_particles.csv')
                        generate_trajectory_video(image_stack, tracked_data, output_path + 'particle_trajectories.avi')
