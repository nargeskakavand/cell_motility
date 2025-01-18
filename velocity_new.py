# import libraries
import pandas as pd
import math

# defining the results parameters
results = {'Particle': [],
           'Distance': [],
           'dx/dt': [],
           'dy/dt': [],
           'frames_count': [],
           'dx/(dt*fc)': [],
           'dy/(dt*fc)': [],
           }

for experiment_round in [1, 2, 3, 4]:
    for sample_type in ['HA452', 'HA3107']:
        for day in [1, 2, 3, 4]:
            for sample_num in [1, 2, 3, 4]:
                for tech_num in [1, 2]:
                    # Define paths for input and output
                    base_path = f"F:/perturbation_all_rounds/velocity/{experiment_round}/" \
                                f"{sample_type}/{day}/{sample_num}/"
                    print(f"Processing folder: {experiment_round}_{day}_{sample_type}_{sample_num}_T{tech_num}")
                    df_results = pd.DataFrame(results)
                    filter_file = base_path + f"T{tech_num}filtered_trajectories.csv"
                    df_filter = pd.read_csv(filter_file)
                    particles = list(df_filter['particle'].unique())
                    count = 0
                    break_check = 0
                    for p in range(len(particles)):
                        dlist = []
                        first_frame = df_filter['frame'][count]
                        first_x = df_filter['x'][count]
                        first_y = df_filter['y'][count]
                        dxt_total = 0
                        dyt_total = 0
                        while df_filter['particle'][count] == particles[p]:
                            if count + 1 < len(df_filter['particle']):
                                if df_filter['particle'][count + 1] == particles[p]:
                                    dx = df_filter['x'][count + 1] - df_filter['x'][count]
                                    dy = df_filter['y'][count + 1] - df_filter['y'][count]
                                    dt = (df_filter['frame'][count + 1] - df_filter['frame'][count]) / 16
                                    dxt = dx / dt
                                    dyt = dy / dt
                                    dxt_total += dxt
                                    dyt_total += dyt
                                    d = math.sqrt((dx * dx) + (dy * dy))
                                    dlist.append(d)
                                count += 1
                            else:
                                break_check = 1
                                break
                        if break_check == 0:
                            last_frame = df_filter['frame'][count - 1]
                            dif_frame = last_frame - first_frame
                        else:
                            last_frame = df_filter['frame'][count]
                            dif_frame = last_frame - first_frame
                        distance = sum(dlist)
                        new_row = {'Particle': particles[p],
                                   'Distance': distance,
                                   'dx/dt': dxt_total,
                                   'dy/dt': dyt_total,
                                   'frames_count': dif_frame,
                                   'dx/(dt*fc)': dxt_total / dif_frame,
                                   'dy/(dt*fc)': dyt_total / dif_frame,
                                   }
                        # append a row to the dataframe to add the results of the current test
                        df_results = pd.concat([df_results, pd.DataFrame.from_records([new_row])])
                    mean_row = {'Particle': 'mean',
                                'Distance': df_results['Distance'].mean(),
                                'dx/dt': df_results['dx/dt'].mean(),
                                'dy/dt': df_results['dy/dt'].mean(),
                                'frames_count': df_results['frames_count'].mean(),
                                'dx/(dt*fc)': df_results['dx/(dt*fc)'].mean(),
                                'dy/(dt*fc)': df_results['dy/(dt*fc)'].mean(),
                                }
                    std_row = {'Particle': 'std',
                               'Distance': df_results['Distance'].std(),
                               'dx/dt': df_results['dx/dt'].std(),
                               'dy/dt': df_results['dy/dt'].std(),
                               'frames_count': df_results['frames_count'].std(),
                               'dx/(dt*fc)': df_results['dx/(dt*fc)'].std(),
                               'dy/(dt*fc)': df_results['dy/(dt*fc)'].std(),
                               }
                    df_results = pd.concat([df_results, pd.DataFrame.from_records([mean_row])])
                    df_results = pd.concat([df_results, pd.DataFrame.from_records([std_row])])
                    file_results = base_path + f"T{tech_num}_velocity_new.csv"
                    df_results.to_csv(file_results, index=False)

