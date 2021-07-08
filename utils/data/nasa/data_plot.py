import matplotlib.pyplot as plt

def plot_df(df, with_anomalies=False, known_anomalies=[], anomalies=[]):
    # line plots
    # line plot for each variable against timestep
    
    if with_anomalies:
        anomalies = df.loc[df['anomaly'] == True]
        df = df.drop(columns=['anomaly'])
    
    n_features = len(df.columns)
    plt.figure(figsize=(30, 5 * n_features))
    plot_i = 0
    for i in range(len(df.columns)):
        name = df.columns[i]
        plt.subplot(len(df.columns), 1, plot_i+1)
        plt.plot(df[name])
        if with_anomalies:
            plt.scatter(anomalies.index, anomalies[name], marker='o', color="green")
        for j in range(len(known_anomalies)):
            t1 = known_anomalies[j][0]
            t2 = known_anomalies[j][1]
            plt.axvspan(t1, t2, color='r', alpha=0.2)
        for j in range(len(known_anomalies)):
            t1 = anomalies[j][0]
            t2 = anomalies[j][1]
            plt.axvspan(t1, t2, color='g', alpha=0.2) 
        plt.title(name, y=0)
        plot_i += 1
    plt.show()

def plot_np(data, x_size=30, y_size=50, known_anomalies=[], anomalies=[]):
    # line plots
    # line plot for each variable against timestep
    n_features = data.shape[1]
    plt.figure(figsize=(x_size, y_size))
    for i in range(n_features):
        plt.subplot(n_features, 1, i+1)
        plt.plot(data[:,i])
        for i in range(len(known_anomalies)):
            t1 = known_anomalies[i][0]
            t2 = known_anomalies[i][1]
            plt.axvspan(t1, t2, color='r', alpha=0.2)
        for i in range(len(anomalies)):
            t1 = anomalies[i][0]
            t2 = anomalies[i][1]
            plt.axvspan(t1, t2, color='g', alpha=0.2)
        
    plt.show()

    
def plot_reconstruction(datasets, x_size=30, y_size=50):
    n_features = datasets[0].shape[0]
    plt.figure(figsize=(x_size, y_size))
    for i in range(n_features):
        plt.subplot(n_features, 1, i+1)
        for d in datasets:
            plt.plot(d[i,:])
    plt.show()