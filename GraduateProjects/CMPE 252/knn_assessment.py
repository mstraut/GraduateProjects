import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.collections import LineCollection

data_pdmp1_file = 'CMPE 252/project_252/files/Data_Challenge_PHM2022_training_data/data_pdmp1.csv'
def format_data(file):
    """
    Removes NaN values and sorts columns
    :param file: path to csv file
    :return: DataFrame containing data after cleaning
    """
    data = []
    # open file and read each line
    with open(file) as openFile:
        for line in openFile:
            # formats into lists
            data_split = line.split(',')
            data_list = [float(x) for x in data_split[1:]]
            data_list.insert(0, int(data_split[0]))  # places fault int value at front of list
            data.append(data_list)
    temp_df = pd.DataFrame(data).rename(columns={0: "Fault Value"})
    temp_df = temp_df.dropna(axis='columns')  # drops all columns with nan
    temp_df = temp_df.rename(columns={0: "Fault Value"})
    return temp_df


def plot_faults(data_df):
    """
    Plots fault time series data
    :param data_df: DataFrame of data
    :return: shows time series plot
    """
    faults = data_df['Fault Value'].unique()
    # dictionary to identify faults
    fault_key = {1: 'No - fault', 2: 'Thicker drill steel',
                 3: 'A - seal missing. Leakage \n from high pressure channel to control channel',
                 4: 'B - seal missing. Leakage \nfrom control channel to return channel.',
                 5: 'Return accumulator, damaged', 6: 'Longer drill steel', 7: 'Damper orifice is larger than usual',
                 8: 'Low flow to the damper circuit', 9: 'Valve damage.A small wear - flat on one of the valve lands',
                 10: 'Orifice on control line outlet larger than usual',
                 11: 'Charge level in high pressure accumulator is low.'}
    # for loop to graph each fault
    for fault_int in faults:
        title = "Fault value: " + fault_key[fault_int]
        fault_df = data_df.where(data_df['Fault Value'] == fault_int).dropna(how='all')
        # setup for multi line graph
        x_max = np.arange(1, len(fault_df.columns))
        y_vals = fault_df.iloc[:,1:].to_numpy()

        y_mean = fault_df.iloc[:,1:].mean().to_numpy()
        y_min = fault_df.iloc[:, 1:].min().to_numpy()
        y_max = fault_df.iloc[:, 1:].max().to_numpy()
        y_ext_vals = [y_max, y_mean, y_min]

        # graphs of mean line segments
        # plt.plot(x_max, y_mean, '-')
        # plt.title(title)
        # plt.show()

        segs = [np.column_stack([x_max, y1]) for y1 in y_ext_vals]

        fig, ax = plt.subplots()
        ax.set_xlim(np.min(x_max), np.max(x_max))
        ax.set_ylim(np.min(y_vals), np.max(y_vals))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        line_segments = LineCollection(segs, array=x_max,
                                       linewidths=(1.5, 1.5, 1.5, 2), colors=colors,
                                       linestyles='solid')
        ax.add_collection(line_segments)
        ax.set_title(title)
        plt.sci(line_segments)
        plt.show()

def knn_accuracy(sensor_name, sensor_df):
    """
    :param sensor_name: string of sensor name
    :param sensor_df: Dataframe with sensor data
    :return: print statement for knn accuracy
    """
    X = sensor_df.drop('Fault Value', axis=1)
    y = sensor_df['Fault Value']
    knn_num = 0
    accuracy = 0
    best_knn = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    # while loop just to check for the best within 10 and under nearest neighbors
    while knn_num <= 10:
        knn_num = knn_num + 1
        knn = KNeighborsClassifier(n_neighbors=knn_num)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy_temp = accuracy_score(y_test, y_pred)
        if accuracy_temp > accuracy:
            accuracy = accuracy_temp
            best_knn = knn_num

    print(best_knn, 'nearest neighbors \t - \t', sensor_name, ' \t- \t'f'Accuracy: {accuracy * 100:.2f}%')



# using pdmp as proof of concept
data_pdmp1 = format_data(data_pdmp1_file)

plot_faults(data_pdmp1)

# data for sensor using KNN for predictions

knn_accuracy('PDMP Sensor', data_pdmp1)

