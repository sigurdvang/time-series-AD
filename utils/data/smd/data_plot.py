import matplotlib.pyplot as plt

def plot_df(df, known_anomalies=[], anomalies=[]):
    # get different colors for adjacent plots
    colors = plt.rcParams["axes.prop_cycle"]()
    
    n_features = len(df.columns)
    plt.figure(figsize=(30, 5 * n_features))
    for i in range(n_features):
        color = next(colors)["color"]
        col = df.columns[i]
        plt.subplot(n_features, 1, i + 1)
        plt.plot(df[col], color=color)
        if len(known_anomalies) > 0:
            for j in range(len(known_anomalies['start'])):
                t1 = known_anomalies['start'][j]
                t2 = known_anomalies['end'][j]
                plt.axvspan(t1, t2, color='r', alpha=0.2)
        if len(anomalies) > 0:
            for j in range(len(anomalies)):
                t1 = anomalies['start'][j]
                t2 = anomalies['end'][j]
                plt.axvspan(t1, t2, color='g', alpha=0.2) 
        plt.title(col, y=0)
    plt.show()

    
def plot_reconstruction(datasets, x_size=30, y_size=50):
    n_features = datasets[0].shape[0]
    plt.figure(figsize=(x_size, y_size))
    for i in range(n_features):
        plt.subplot(n_features, 1, i+1)
        for d in datasets:
            plt.plot(d[i,:])
    plt.show()
    
def plot_np(data, x_size=30, y_size=50, known_anomalies=[], anomalies=[]):
    # line plots
    # line plot for each variable against timestep
    n_features = data.shape[1]
    plt.figure(figsize=(x_size, y_size))
    colors = plt.rcParams["axes.prop_cycle"]()
    for i in range(n_features):
        color = next(colors)["color"]
        plt.subplot(n_features, 1, i+1)
        plt.plot(data[:,i], color=color)
        if len(known_anomalies) > 0:
            for j in range(len(known_anomalies['start'])):
                t1 = known_anomalies['start'][j]
                t2 = known_anomalies['end'][j]
                plt.axvspan(t1, t2, color='r', alpha=0.2)
        if len(anomalies) > 0:
            for j in range(len(anomalies)):
                t1 = anomalies['start'][j]
                t2 = anomalies['end'][j]
                plt.axvspan(t1, t2, color='g', alpha=0.2) 
        
    plt.show()