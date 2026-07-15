import argparse
from pathlib import Path

import networkx as nx
from sklearn.metrics import adjusted_rand_score

from datasets.GraphProcessor import GraphProcessor
from models.FE_SyncMap import SyncMap
from utils.draw_3d import *
from utils.utils import Metrics
from utils.utils import set_seed


def load_graph(dot_path):
    g = nx.DiGraph(nx.nx_agraph.read_dot(dot_path))
    idx = {n: i for i, n in enumerate(g.nodes())}
    A = np.zeros((len(idx), len(idx)), dtype=np.float32)
    for u, v, d in g.edges(data=True):
        A[idx[u], idx[v]] = float(d.get("weight", 1.0))
    return A, idx


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="nlp", help='Run which data.')
    parser.add_argument('--time_delay', type=int, default=10, help='Set the time delay.')
    parser.add_argument('--task_type', type=int, default=1,
                        help='Choose the task to run the model, e.g., 1 for GraphWalkTest, 2 for FixedChunkTest, 3 for GraphWalkTest with sequence2.dot, 4 for GraphWalkTest with sequence1.dot, 5 for LongChunkTest, 6 for OverlapChunkTest1, 7 for OverlapChunkTest2')
    parser.add_argument('--sequence_length', type=int, default=80_0000,
                        help='Set the sequence length to run the model.')
    parser.add_argument('--iter', type=int, default=1, help='Training time for each data.')
    parser.add_argument('--map_dimensions', type=int, default=15, help='Set the dimension of the map.')
    parser.add_argument('--adaptation_rate', type=float, default=0.001, help='Set the adaptation rate.')
    parser.add_argument('--seed', type=int, default=412, help='Random seed.')
    parser.add_argument('--draw', type=int, default=1, help='Draw the image.')

    parser.add_argument('--eps', type=float, default=4.5, help='Fixed eps value for DBSCAN clustering.')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    root_dir = Path("data")
    if args.dataset == "all":
        dot_files = list(root_dir.rglob("*.dot"))
        num = len(dot_files)

        score_all = 0.
        for dot_path in dot_files:
            # print("dot_path:", dot_path)
            processor = GraphProcessor(args.time_delay, dot_path)

            output_size = processor.getOutputSize()
            input_sequence, input_class = processor.getSequence(args.sequence_length)

            # 2. Train and Test
            number_of_nodes = output_size
            neuron_group = SyncMap(number_of_nodes, args.map_dimensions, args.adaptation_rate * output_size)

            neuron_group.input(input_sequence)
            labels = neuron_group.organize()

            ari = adjusted_rand_score(labels, processor.trueLabel())
            score_all += ari
            print("Data:", dot_path)
            # print("Learned Labels: ", labels)
            # print("Correct Labels: ", env.trueLabel())
            print("ARI  =", ari)

        score_avg = score_all / num
        print("score_avg:", score_avg)

    else:
        file_path = r"E:\SyncMap\FE-SyncMap\the-SyncMap-Zoo\data\chain_mixed5_120.dot"
        processor = GraphProcessor(args.time_delay, file_path, state_memory=4)
        output_size = processor.getOutputSize()
        true_labels = processor.trueLabel()

        theta0, input_seq_trajectory, input_seq_onehot = processor.first_firing_phase(args.sequence_length)

        number_of_nodes = output_size
        number_of_nodes = output_size
        effective_lr = args.adaptation_rate * number_of_nodes
        print(f"[lr] base={args.adaptation_rate}, n={number_of_nodes}, effective_alpha={effective_lr}")
        syncmap = SyncMap(number_of_nodes,
                          args.map_dimensions,
                          effective_lr)
        # 2. Generate input sequence
        input_seq_trajectory, input_seq_onehot = processor.random_walk_on_graph(args.sequence_length)
        print("input_seq_trajectory:", input_seq_trajectory.shape)  # (100000,)
        print("input_seq_onehot:", input_seq_onehot.shape)  # (100000, 30)

        # generate input sequence for syncmap   add memory add more activated nodes
        input_seq = processor.generate_memory_sequence(input_seq=input_seq_onehot)
        print("input_seq:", input_seq.shape)  # (100000, 30)
        print("input_seq:", input_seq)

        true_counts_per_row = np.sum(input_seq, axis=1)
        print(true_counts_per_row)

        # 3. Train and Test
        print("training SyncMap...")
        syncmap.input(input_seq, r_seq=input_seq_trajectory)

        learned_map, draw_3d_nodes, plus, minus = syncmap.get_syncmap(isMovMean=False)
        print("learned_map:", learned_map)  # (30, 2)
        print("draw_3d_nodes:", draw_3d_nodes.shape)  # (99990, 30, 2)
        print("plus:", plus.shape)  # (99990, 2)
        print("minus:", minus.shape)  # (99990, 2)

        # 3. 计算度量并用固定 eps 做聚类
        readout = Metrics(learned_map, ground_truth=true_labels)

        # ★ 用固定 eps（来自命令行参数）
        fixed_eps = args.eps
        readout.dbscan_(eps=fixed_eps, min_samples=1, print_result=False)
        nmi = readout.cal_NMI(print_result=False)

        print(f"DBSCAN with fixed eps = {fixed_eps}")
        print(f"NMI = {nmi}")
        print("Predicted labels:", readout.predicted_labels)

        print("record_traj =", getattr(syncmap, "record_traj", None))
        print("viz_stride  =", getattr(syncmap, "viz_stride", None))
        print("frames(coords)=", len(draw_3d_nodes),
              "frames(plus)=", len(plus),
              "frames(minus)=", len(minus))

        if args.draw == 1:
            print("true:", true_labels)
            color = labels2colors(true_labels)
            print("color:", color)
            hex = convert_rgb_list_to_hex(color)
            print("hex:", hex)
            # create_scatter_gif_3d(draw_3d_nodes, hex)

            if args.map_dimensions == 3:
                animate_3d_coords(draw_3d_nodes, hex)

            else:
                # animate_2d_coords_stride(draw_3d_nodes, hex)
                animate_2d_coords_center(draw_3d_nodes, hex, plus, minus)
