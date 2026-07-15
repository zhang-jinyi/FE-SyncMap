import random

import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, maxdists
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def plot(syncmap, color=None, save=False, filename="plot_map.png"):
    ax = plt.scatter(syncmap[:, 0], syncmap[:, 1], c='b')

    if save == True:
        plt.savefig(filename)

    plt.show()
    plt.close()

def plot_with_labels(syncmap, labels, color=None,                 # ← 新增 labels
         save=True, filename="plot_map.png",
         marker_size=80, fontsize=11):
    """
    syncmap : (N, 2) ndarray      已学习到的 2D 坐标
    labels  : list[str] length=N  每个节点要显示的字母或单词
    color   : None / list         与 plt.scatter 的 c 参数兼容
    """
    syncmap = np.asarray(syncmap)
    N = syncmap.shape[0]
    print("len(labels):", len(labels))
    print("N:", N)
    assert len(labels) == N, "labels length must equal number of nodes"

    plt.figure(figsize=(6, 6))
    plt.scatter(syncmap[:, 0], syncmap[:, 1],
                s=marker_size,  # 圈大小
                facecolors="#ffd60a",  # 亮黄
                edgecolors="black",  # 外框黑
                linewidths=1.0,
                zorder=2)

    # 给每个点加文字
    for i, (x, y) in enumerate(syncmap):
        plt.text(x, y, labels[i],
                 ha='center', va='center',
                 fontsize=fontsize, color='black', zorder=3)

    # 让坐标轴保持比例 & 网格
    margin = 0.5
    R = np.abs(syncmap).max() + margin
    plt.xlim(-R, R); plt.ylim(-R, R)
    plt.gca().set_aspect('equal'); plt.grid(True)
    plt.title("SyncMap 2D projection with labels")

    if save:
        plt.savefig(filename, dpi=100)
        print(f"figure saved to {filename}")

    plt.show()
    plt.close()

# def to_categorical(y, num_classes):
#     out = np.zeros(num_classes)
#     out[y] = 1
#     return out
def to_categorical(x, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with `categorical_crossentropy`.

    Args:
        x: Array-like with class values to be converted into a matrix
            (integers from 0 to `num_classes - 1`).
        num_classes: Total number of classes. If `None`, this would be inferred
            as `max(x) + 1`. Defaults to `None`.

    Returns:
        A binary matrix representation of the input as a NumPy array. The class
        axis is placed last.

    Example:

    >>> a = to_categorical([0, 1, 2, 3], num_classes=4)
    >>> print(a)
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    """
    x = np.array(x, dtype="int64")
    input_shape = x.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    x = x.reshape(-1)
    if not num_classes:
        num_classes = np.max(x) + 1
    batch_size = x.shape[0]
    categorical = np.zeros((batch_size, num_classes))
    categorical[np.arange(batch_size), x] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def show_graph(save=False):
    path = "../data/chain_mixed120_5.dot"

    G = nx.DiGraph(nx.nx_agraph.read_dot(path))

    options = {
        'node_size': 100,
        'arrowstyle': '-|>',
        'arrowsize': 12,
    }
    nx.draw_networkx(G, arrows=True, **options)

    if save == True:
        plt.savefig("./results/graph_plot.png")

    plt.show()


class Metrics:
    def __init__(self, input_map=None, input_matrix=None, ground_truth=None):
        self.input_map = input_map
        self.input_matrix = input_matrix
        self.ground_truth = ground_truth
        self.predicted_labels = None
        self.NMI = None

    def cal_NMI(self, print_result=True):
        if self.predicted_labels is None:
            print("No predicted labels found. Run dbscan_() first.")
            return None
        # Calculate NMI
        self.NMI = normalized_mutual_info_score(self.ground_truth, self.predicted_labels)
        if print_result:
            print("NMI: ", self.NMI)
        return self.NMI

    def dbscan_(self, map=None, eps=0.1, min_samples=2, print_result=True):
        if map is None:
            map = self.input_map
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(map)
        self.predicted_labels = clustering.labels_
        if print_result:
            print("DBSCAN clustering done. Data updated.")
            print("Predicted labels: ", self.predicted_labels)
            print("Ground truth: ", self.ground_truth)
        return self.predicted_labels

    def hierarchical_organize(self, map=None, hierarchy=None, method='ward', print_result=True):
        if map is None:
            map_ = self.input_map
        else:
            map_ = map
        input_size = map_.shape[0]
        # method = "single"
        Z = linkage(map_, method)
        # fig = plt.figure(dpi=150)
        # label_list = [i for i in range(1, self.input_size+1)]
        # dendrogram(Z, color_threshold=0, above_threshold_color='k', labels=label_list)
        # dendrogram(Z, labels=label_list)

        Z_maxdists = maxdists(Z)
        d_diff_list = []
        for d in range(len(Z_maxdists) - 1):
            d_diff = Z_maxdists[d + 1] - Z_maxdists[d]
            d_diff_list.append(d_diff)

        d_diff_index = np.argsort(d_diff_list)[::-1]

        max_diff = d_diff_index[0]
        tmp_d_diff_index = [max_diff]
        for d in d_diff_index[1:]:
            if max_diff > d:
                max_diff = d
                tmp_d_diff_index.append(d)
        d_diff_index = tmp_d_diff_index

        total_hierarchy = len(d_diff_index)
        if hierarchy is not None:
            total_hierarchy = hierarchy

        labels = np.empty((total_hierarchy, input_size), dtype=int)
        for h in range(total_hierarchy):
            label = [-1 for _ in range(input_size)]
            if h < len(d_diff_index):
                n_cluster = input_size - d_diff_index[h] - 1
                label = AgglomerativeClustering(n_clusters=n_cluster, linkage=method).fit_predict(map_)
            labels[h, :] = label

        # self.labels = np.flip(labels, axis=0)
        self.labels = labels
        self.Z_linkage = Z
        print("Hierarchical clustering done. Data updated.")
        if print_result:
            print("Ground truth: ", self.ground_truth)
            print("Labels: ", self.labels)

        return self.labels

    def plot_dendrogram(self, Z=None, isPlotLabel=False, labels=None):
        if Z is None:
            Z = self.Z_linkage
            if Z is None:
                Z = linkage(self.input_map, 'ward')
        if labels is None:
            labels = self.labels

        fig = plt.figure(dpi=150)
        label_list = [i for i in range(1, labels.shape[1] + 1)]
        if isPlotLabel:
            label_list = labels[0, :]
            dendrogram(Z, color_threshold=0, above_threshold_color='k', labels=np.array(label_list))
        else:
            dendrogram(Z)
        plt.show()
        return None

def draw_matrix(data_vis, max_rows = 5000):
    if max_rows and data_vis.shape[0] > max_rows:
        data_vis = data_vis[:max_rows]
    else:
        data_vis = data_vis

    print("data_vis:", data_vis.shape)#(5000, 60)
    fig, ax = plt.subplots(figsize=(8, max(4, data_vis.shape[0] * 0.004)))
    cmap = matplotlib.colors.ListedColormap(['red', 'blue'])  # False→red, True→blue
    ax.imshow(data_vis, aspect='auto', interpolation='nearest', cmap=cmap)

    ax.set_xlabel('Column index')
    ax.set_ylabel('Row index')
    ax.set_title('input_seq_onehot\n(True→blue,False→red)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_graph()
