"""
author: Ali Shafiezadeh

event_recognition: This code finds the first heel strike and foot off times during stair descent, downhill walking, and
level walking using FFA and residual methods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import os

# Read the raw data set (the data set used in this code is the CAMS data set)
files = [file for file in os.listdir('data set') if file.endswith('.csv')]


def read_data(file_id):
    """
    Read the .csv file and obtain the events included in the file.

    :param file_id: raw data set, the file in which we wish to find the times.
    :return: read data and the properties of it (cache)
    """
    # Each raw data contains some important times at the beginning of the file. We need those times.
    x = 'NaN'
    while x[0] != leg + ' Foot Strike':
        x = file_id.readline()
        x = x.split(',')
    t_HS_1 = float(x[1])

    # For the stair descent task we already have the second heel strike.
    if 'sd' in trial:
        x = 'NaN'
        while x[0] != leg + ' Foot Strike':
            x = file_id.readline()
            x = x.split(',')
        t_HS_2 = float(x[1])

    if leg == 'Left':
        leg_temp = 'Right'
    else:
        leg_temp = 'Left'
    x = 'NaN'
    while x[0] != leg_temp + ' Foot Off':
        x = file_id.readline()
        x = x.split(',')
    t_FO_2 = float(x[1])

    x = 'NaN'
    while x[0] != 'TIME':
        x = file_id.readline()
        x = x.split(',')

    # Read data
    headers_csv = x
    labels_csv = file_id.readline().split(',')
    file_id.readline()

    data_csv = pd.read_csv(file_id)
    data_csv.apply(pd.to_numeric)

    # Convert data to DataFrame with specified columns
    csv_file = pd.DataFrame(np.array(data_csv), columns=labels_csv)
    file_id.close()

    # Read_data cache
    cache = {'data_csv': data_csv,
             'labels_csv': labels_csv,
             'headers_csv': headers_csv,
             't_HS_1': t_HS_1,
             't_FO_2': t_FO_2,
             'leg_temp': leg_temp}
    if 'sd' in trial:
        cache['t_HS_2'] = t_HS_2

    return csv_file, cache


def read_marker(csv_file):
    """
    Read the toe marker data and TIME

    :param csv_file: csv_file which is the read data from read_data function
    :return: toe marker data and TIME data
    """
    coordinates = ["X", "Y", "Z"]

    # Read TIME data and initialize marker_data
    marker_data = pd.DataFrame()
    TIME = csv_file.iloc[:, csv_file.columns.get_level_values(0) == "time"]

    # Read the toe marker data
    for j in coordinates:
        marker_data[f"{j}"] = csv_file.iloc[:, csv_file.columns.get_level_values(0) == f"{toe_marker}_{j}"]

    # For right leg subjects the motion is reversed and we should multiply it to -1
    if leg == 'Right':
        marker_data["Y"] = -marker_data["Y"]

    return marker_data, TIME


def residual_filters(Y, fs=100):
    """
    Apply butterworth 4th order low frequency with different cut-off frequencies on the marker data and save the results
    of each frequency

    :param Y: Toe marker data from the read_marker function
    :param fs: Sample rate
    :return: Results of each cut-off frequency saved in a data frame
    """

    # Initialize Y_filtered data frame
    Y_filtered = pd.DataFrame()
    frequencies = list(map(lambda x: round(x, 1), np.arange(1, 10, 0.1)))

    # Apply butterworth filter and save the result of it
    for frequency in frequencies:
        w = frequency / (fs / 2)
        b, a = sg.butter(4, w, 'low')
        Y_filtered[f"{frequency}Hz"] = sg.filtfilt(b, a, Y)

    return Y_filtered


def residual_plot(Y_filtered, Y):
    """
    Compute the best cut-off frequency

    :param Y_filtered: The output of residual_filters function, which is a dictionary containing results of each cut-off
     frequency
    :param Y: The marker data
    :return: Plot residual method results and return the best cut-off frequency
    """

    dev = np.array(Y).reshape((len(Y_filtered), 1)) - np.array(Y_filtered)
    RMSE = np.sqrt(np.sum(np.multiply(dev, dev), axis=0) / len(dev))
    t = np.array(list(map(lambda x: float(format(x, ".1f")), np.arange(1, 10, 0.1))))

    t1 = int(np.where(t==5)[0])
    y2, x2, y1, x1 = RMSE[-1], t[-1], RMSE[t1], t[t1]
    a = (y2 - y1) / (x2 - x1)
    b = y2 - a * x2
    y0 = b

    x = np.array(list(map(lambda x: float(format(x, ".1f")), np.arange(0, 10, 0.1))))
    y = a * x + b

    f_temp = np.argmin(RMSE - y0)
    f = t[f_temp]

    # plt.plot(t, RMSE)
    # plt.plot(x, y, 'r--')
    # plt.title("Residual graph")
    # plt.xlabel("Frequency(Hz)")
    # plt.ylabel("RMSE(mm)")
    # plt.show()
    #
    # Y_t = np.linspace(0, len(Y), len(Y))
    #
    # plt.plot(Y_t, Y_filtered[f"{f}Hz"], label='filtered')
    # plt.plot(Y_t, Y, label='raw')
    # plt.title("Filtered data")
    # plt.xlabel("Frequency(Hz)")
    # plt.ylabel(f"{marker_name}_y(mm)")
    # plt.legend()
    # plt.show()

    return f


def derivative_propagate(Y, delta_t=0.01):
    """
    Derivative propagate; compute velocity, acceleration, and jerk of the marker data.

    :param Y: The marker data
    :param delta_t: Sample rate time
    :return: Velocity, acceleration, and jerk of the marker data
    """
    temp2 = Y[1:]
    temp1 = Y[:-1]
    V = (temp2 - temp1) / delta_t

    temp2 = V[1:]
    temp1 = V[:-1]
    A = (temp2 - temp1) / delta_t

    temp2 = A[1:]
    temp1 = A[:-1]
    J = (temp2 - temp1) / delta_t

    return V, A, J


def FO_rec(marker, csv_cache, fs=100):
    """
    Find "foot off" time using previous functions

    :param marker: The marker data
    :param csv_cache: Cache of the .csv file, output of the read_data function
    :param fs: Sample rate
    :return: "Foot off" time, t_FO cache
    """
    # 1 Filter the marker data with different cut-off frequencies
    X = marker["Y"].dropna()
    X_filtered = residual_filters(X, 100)

    # 2 Calculate the best cut-off frequency
    f = residual_plot(X_filtered, X)
    # print(f"Cut-off frequency: {f}Hz")

    # Filter the marker data
    X_filtered = np.array(X_filtered[f"{f}Hz"]).reshape((len(X), 1))

    # Calculate velocity, acceleration, and jerk
    V_X, A_X, J_X = derivative_propagate(X_filtered)

    # Plot results
    ## X_filtered_normalized = X_filtered / (max(X_filtered) - min(X_filtered))
    ## X_t = np.linspace(0, len(X), len(X))
    ##
    ## plt.plot(X_t, X_filtered_normalized, label='X')
    ##
    ## t_A = np.linspace(0, len(A), len(A))
    ## t_J = np.linspace(0, len(J), len(J))
    ##
    ## A_normalized = A / (max(A) - min(A))
    ## J_normalized = J / (max(J) - min(J))
    ##
    ## plt.plot(t_A, A_normalized, label='A_X')
    ## plt.plot(t_J, J_normalized, label='J_X')
    ## plt.legend()
    ## plt.show()

    # Read the first heel strike time from csv_cache
    t_HS = int(csv_cache['t_HS_1'] * fs)

    # In a window of 50Hz find the minimum of A
    t_window = [t_HS - fs//2, t_HS + fs//2]
    A_window = A_X[t_window[0]:t_window[1]]
    A_min = min(A_window)
    t_A_min = int(np.where(A_X == A_min)[0])

    # Calculate the foot off frequency
    t1 = t_A_min - 1
    t2 = t_A_min + 1
    J1 = J_X[t1]
    J2 = J_X[t2]
    t_FO = int(t1 + J1 / (J1-J2)) / 100
    cache = {'t_A_min': t_A_min,
             't1': t1,
             't2': t2,
             'J1': J1,
             'J2': J2}

    return t_FO, cache


def HS_rec(csv_file, csv_cache, fs=100):
    """
    Find "foot off" time using previous functions

    :param csv_file: The read data, output of the read_data function
    :param csv_cache: Cache of the read_data function
    :param fs: Sample rate
    :return: Second heel strike in the down hill walking tasks and the cache
    """

    # The code in this function is similar to FO_rec
    coordinates = ["X", "Y", "Z"]

    marker_data = pd.DataFrame()
    for j in coordinates:
        marker_data[f"{j}"] = csv_file.iloc[:, csv_file.columns.get_level_values(0) == f"{heel_marker}_{j}"]

    Z = marker_data["Z"].dropna()

    Z_filtered = residual_filters(Z, 100)

    f_Z = residual_plot(Z_filtered, Z)
    # print(f"Cut-off frequency: {f}Hz")

    Z_filtered = np.array(Z_filtered[f"{f_Z}Hz"]).reshape((len(Z), 1)).reshape(-1, 1)
    V_Z, A_Z, J_Z = derivative_propagate(Z_filtered)

    # Z_filtered_normalized = Z_filtered / (max(Z_filtered) - min(Z_filtered))
    # Z_t = np.linspace(0, len(Z), len(Z))
    #
    # plt.plot(Z_t, Z_filtered_normalized, label='Z')
    #
    # t_A = np.linspace(0, len(A_Z), len(A_Z))
    # t_J = np.linspace(0, len(J_Z), len(J_Z))
    #
    # A_normalized_Z = A_Z / (max(A_Z) - min(A_Z))
    # J_normalized_Z = J_Z / (max(J_Z) - min(J_Z))
    #
    # plt.plot(t_A, A_normalized_Z, label='A_Z')
    # plt.plot(t_J, J_normalized_Z, label='J_Z')
    # plt.legend()
    # plt.show()

    t_FO = int(csv_cache['t_FO_2'] * fs)
    t_window = [t_FO - fs, t_FO]
    t_HS = np.where(Z_filtered == np.min(Z_filtered[t_window[0]:t_window[1]]))

    t_window2 = [int(t_HS[0]) - fs//10, int(t_HS[0]) + fs//10]
    A_window = A_Z[t_window2[0]:t_window2[1]]
    A_max = np.max(A_window)
    t_A_max = int(np.where(A_Z == A_max)[0])

    t1 = t_A_max - 1
    t2 = t_A_max + 1
    J1 = J_Z[t1]
    J2 = J_Z[t2]
    t_HS = int(t1 + J1 / (J1 - J2)) / 100
    cache_Z = {'t_A_min': t_A_max,
             't1': t1,
             't2': t2,
             'J1': J1,
             'J2': J2}

    assert t_HS < csv_cache['t_FO_2']

    return t_HS, cache_Z


T = {'trial': [],
     't_HS_1': [],
     't_FO_1': [],
     't_HS_2': [],
     't_FO_2': []}
for file in files[0:]:
    # Apply the functions above on each .csv file in the data set folder

    file_split = file.split('_')
    trial = file_split[5]
    subject = "_".join([file_split[0], file_split[5], file_split[6]])
    T['trial'].append(subject)

    file_id = open("data set/"+file, "r")

    # Finding the dominant leg; it indicates the cycle leg
    if file.split('_')[0][2] == 'R':
        leg = 'Right'
    elif file.split('_')[0][2] == 'L':
        leg = 'Left'
    else:
        raise Exception("Leg did not matched!")

    # Defining the toe marker according to cycle leg
    if leg == 'Right':
        toe_marker = 'LTTO'
        heel_marker = 'RTHL'
    else:
        toe_marker = 'RTTO'
        heel_marker = 'LTHL'

    # Reading data
    csv_file, cache = read_data(file_id)
    marker, TIME = read_marker(csv_file)

    # Call HS_rec function to calculate second heel strike in down hill walking
    if trial == 'rd':
        t_HS, HS_cache = HS_rec(csv_file, cache)
        cache['t_HS_2'] = t_HS

    # Append times to the T dictionary
    T['t_HS_1'].append(cache['t_HS_1'])
    T['t_HS_2'].append(cache['t_HS_2'])
    T['t_FO_2'].append(cache['t_FO_2'])

    t_FO_1, FO_cache = FO_rec(marker, cache)
    T['t_FO_1'].append(t_FO_1)

    file_id.close()

    # print(f"t_FO = {t_FO}")

# Convert T dictionary to DataFrame
T = pd.DataFrame(T)

# Save T
T.to_csv('GRF_shift_times.csv', index=False)
