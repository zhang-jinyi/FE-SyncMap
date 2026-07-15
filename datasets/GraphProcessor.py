import os
import re
import warnings
from collections import deque

import networkx as nx
import numpy as np


def _strip_quotes(value):
    if value is None:
        return None
    return str(value).strip().strip('"').strip("'")


def _read_dot_robust(path: str) -> nx.Graph:
    """读取 DOT 图；优先使用 pygraphviz，失败时回退到 pydot。"""
    try:
        return nx.nx_agraph.read_dot(path)
    except Exception:
        pass

    try:
        import pydot
    except Exception as exc:
        raise RuntimeError(
            "读取 .dot 文件需要安装 pygraphviz 或 pydot。"
        ) from exc

    graphs = pydot.graph_from_dot_file(path)
    if not graphs:
        raise ValueError(f"读取失败或 DOT 文件为空: {path}")

    pydot_graph = graphs[0]
    directed = (pydot_graph.get_type() or "").lower() == "digraph"
    graph = nx.MultiDiGraph() if directed else nx.MultiGraph()

    for node in pydot_graph.get_nodes():
        name = _strip_quotes(node.get_name())
        if name in ("node", "edge", "graph"):
            continue
        attributes = {
            key: _strip_quotes(value)
            for key, value in (node.get_attributes() or {}).items()
        }
        graph.add_node(name, **attributes)

    for edge in pydot_graph.get_edges():
        source = _strip_quotes(edge.get_source())
        target = _strip_quotes(edge.get_destination())
        attributes = {
            key: _strip_quotes(value)
            for key, value in (edge.get_attributes() or {}).items()
        }
        graph.add_edge(source, target, **attributes)

    return graph


def _read_gml_robust(path: str) -> nx.Graph:
    """读取 GML 图；优先将 label 作为节点标识，失败时使用节点 ID。"""
    try:
        return nx.read_gml(path, label="label")
    except Exception:
        return nx.read_gml(path, label=None)


def _read_graph_any(path: str) -> nx.Graph:
    """根据扩展名读取 DOT/GML，并统一转换为有向图。"""
    extension = os.path.splitext(path)[1].lower()

    if extension == ".dot":
        graph = _read_dot_robust(path)
    elif extension == ".gml":
        graph = _read_gml_robust(path)
    else:
        try:
            graph = _read_gml_robust(path)
        except Exception:
            graph = _read_dot_robust(path)

    if not isinstance(graph, (nx.DiGraph, nx.MultiDiGraph)):
        graph = nx.DiGraph(graph)

    return graph


def _parse_int_or_category_list(values, prefer_int=True) -> np.ndarray:
    """将节点标签转换为从 0 开始的整数类别。"""
    cleaned = []
    all_int = True

    for value in values:
        text = _strip_quotes(value)
        if text is None:
            all_int = False
            cleaned.append(None)
            continue

        match = re.search(r"-?\d+", text)
        if prefer_int and match:
            cleaned.append(int(match.group()))
        else:
            all_int = False
            cleaned.append(text)

    if all_int and cleaned and all(value is not None for value in cleaned):
        labels = np.asarray(cleaned, dtype=np.int64)
        return labels - labels.min()

    category_to_index = {}
    labels = []
    for value in cleaned:
        if value is None:
            labels.append(-1)
            continue
        if value not in category_to_index:
            category_to_index[value] = len(category_to_index)
        labels.append(category_to_index[value])

    return np.asarray(labels, dtype=np.int64)


def _extract_true_labels(graph: nx.Graph) -> np.ndarray:
    """从常见节点属性中提取真实类别标签。"""
    candidates = [
        "true_label",
        "label",
        "club",
        "community",
        "class",
        "gt",
        "category",
        "y",
        "value",
    ]

    for key in candidates:
        if all(key in graph.nodes[node] for node in graph.nodes):
            values = [graph.nodes[node][key] for node in graph.nodes]
            return _parse_int_or_category_list(
                values,
                prefer_int=(key == "label"),
            )

    warnings.warn(
        "未在图中找到常见标签字段（label/club/...），true_label 将填充为 0。"
    )
    return np.zeros(graph.number_of_nodes(), dtype=np.int64)


class GraphProcessor:
    """为 FE-SyncMap 读取图并生成随机游走训练序列。"""

    def __init__(self, time_delay, dataset, state_memory=2):
        # 保留 time_delay 参数以兼容现有训练脚本。
        self.time_delay = time_delay
        self.G = _read_graph_any(str(dataset))
        self.true_label = _extract_true_labels(self.G).astype(np.int64)
        self.output_size = self.G.number_of_nodes()

        adjacency = nx.adjacency_matrix(self.G).todense()
        self.A = np.asarray(adjacency, dtype=np.float64)

        for node_index in range(self.output_size):
            row_sum = self.A[node_index].sum()
            if row_sum == 0:
                raise ValueError(
                    f"节点 {node_index} 没有出边，无法生成图随机游走序列。"
                )
            self.A[node_index] /= row_sum

        self.working_memory = deque(maxlen=state_memory)

    def trueLabel(self):
        return self.true_label

    def getOutputSize(self):
        return self.output_size

    def random_walk_on_graph(self, sequence_size, reset_time=None):
        """按邻接矩阵中的转移概率生成节点轨迹和 one-hot 序列。"""
        connection_matrix = self.A
        num_nodes = connection_matrix.shape[0]
        no_outgoing = np.where(connection_matrix.sum(axis=1) == 0)[0]

        if len(no_outgoing) == num_nodes:
            raise ValueError("图中所有节点都没有出边，无法随机游走。")

        current_node = np.random.choice(num_nodes)
        while current_node in no_outgoing:
            current_node = np.random.choice(num_nodes)

        trajectory = np.empty(sequence_size, dtype=np.int32)
        one_hot_vectors = np.zeros(
            (sequence_size, num_nodes),
            dtype=np.bool_,
        )
        steps_since_reset = 0

        for step in range(sequence_size):
            trajectory[step] = current_node
            one_hot_vectors[step, current_node] = True

            should_reset = (
                reset_time is not None
                and steps_since_reset == reset_time
            )
            if connection_matrix[current_node].sum() == 0 or should_reset:
                current_node = np.random.choice(num_nodes)
                while current_node in no_outgoing:
                    current_node = np.random.choice(num_nodes)
                steps_since_reset = 0
                continue

            probabilities = connection_matrix[current_node]
            probabilities = probabilities / probabilities.sum()
            current_node = np.random.choice(num_nodes, p=probabilities)
            steps_since_reset += 1

        return trajectory, one_hot_vectors

    def first_firing_phase(self, seq_len, T=None):
        """生成随机游走，并计算每个节点首次出现时间对应的相位。"""
        trajectory, vectors = self.random_walk_on_graph(seq_len)

        first_hit = np.full(self.output_size, np.inf)
        for step, node in enumerate(trajectory):
            if np.isinf(first_hit[node]):
                first_hit[node] = step

        visited = first_hit[np.isfinite(first_hit)]
        if visited.size == 0:
            raise ValueError("随机游走序列为空，无法计算首次触发相位。")

        finite_max = visited.max()
        first_hit[np.isinf(first_hit)] = finite_max + 1

        if T is None:
            T = finite_max + 2
        if T <= 0:
            raise ValueError("相位周期 T 必须大于 0。")

        theta = 2 * np.pi * (first_hit % T) / T
        return theta.astype(np.float32), trajectory, vectors

    def generate_memory_sequence(self, input_seq):
        """将最近 state_memory 步的 one-hot 状态合并为联合激活序列。"""
        output_seq = []

        for state in input_seq:
            self.working_memory.append(state)
            current_memory = np.asarray(self.working_memory)
            active_nodes = current_memory.sum(axis=0).astype(np.bool_)
            output_seq.append(active_nodes)

        return np.asarray(output_seq)
