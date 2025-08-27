import numpy as np
import matplotlib.pyplot as plt
from lab_utils import run_kMeans
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap




class plt_one_addpt_onclick_kmeans:
    def __init__(self, X, initial_centroids, max_iters=10):
        self.X = X
        self.centroids = initial_centroids
        self.max_iters = max_iters
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.scatter(X[:, 0], X[:, 1], color='gray', label='Data Points')

        # دکمه اجرا
        ax_button = plt.axes([0.1, 0.05, 0.2, 0.075])
        self.btn_run = Button(ax_button, 'Run K-Means', color='lightblue')
        self.btn_run.on_clicked(self.run_kmeans)

        self.fig.canvas.mpl_connect('button_press_event', self.add_data)

    def add_data(self, event):
        if event.inaxes == self.ax:
            x_new, y_new = event.xdata, event.ydata
            self.X = np.vstack((self.X, [x_new, y_new]))
            self.ax.scatter(x_new, y_new, color='gray')
            self.fig.canvas.draw()

    def run_kmeans(self, event):
        centroids, idx = run_kMeans(self.X, self.centroids, self.max_iters)

        self.ax.clear()
        cmap = ListedColormap(['red', 'blue', 'green'])
        self.ax.scatter(self.X[:, 0], self.X[:, 1], c=idx, cmap=cmap, label='Clustered Data')
        self.ax.scatter(centroids[:, 0], centroids[:, 1], s=200, color='yellow', label='Centroids', marker='X')
        self.ax.set_title("K-Means Clustering")
        self.ax.legend()
        self.fig.canvas.draw()