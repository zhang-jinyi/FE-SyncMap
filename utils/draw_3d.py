import os
import re
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from matplotlib.animation import FuncAnimation, PillowWriter


def labels2colors(labels):
    labels = np.array(labels)
    labels = labels - np.min(labels)
    # if len(labels) <= 10:
    #     colorbar = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'pink', 'brown', 'gray']
    # else:
    if len(np.unique(labels)) <= 20:
        colorbar = sns.color_palette("tab20", len(np.unique(labels)))  # hls
    else:
        colorbar = sns.color_palette("gist_ncar", len(np.unique(labels)))  # hls
    # Convert RGB tuples to Plotly's "rgb(r, g, b)" format
    colorbar = [f'rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})' for r, g, b in colorbar]
    colorbar = np.random.choice(colorbar, len(colorbar), replace=False)
    return [colorbar[label] for label in labels]


def rgb_to_hex(rgb):
    # Extract the integers using regex
    rgb_values = re.findall(r'\d+', rgb)
    # Convert the integers to hex and format them properly
    return "#{:02x}{:02x}{:02x}".format(int(rgb_values[0]), int(rgb_values[1]), int(rgb_values[2]))


def convert_rgb_list_to_hex(rgb_list):
    return [rgb_to_hex(rgb) for rgb in rgb_list]


def save_frame_3d(frame, colors, temp_dir, i, dpi=80, iter_multiplier=1000, xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
                  marker_size=100):
    '''
    Save a single frame of a 3D scatter plot.

    Parameters:
        frame (np.ndarray): The frame data with shape (t, 3).
        colors (list): List of colors for each point.
        temp_dir (str): Directory to save the temporary frame image.
        i (int): The index of the frame.
        dpi (int): Resolution of the output image.
        iter_multiplier (int): Multiplier for iteration count in titles.
        xlim (tuple): Limits for the x-axis.
        ylim (tuple): Limits for the y-axis.
        zlim (tuple): Limits for the z-axis.
    '''
    sns.set_theme()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    for j in range(frame.shape[0]):
        ax.scatter(frame[j, 0], frame[j, 1], frame[j, 2], color=colors[j], s=marker_size)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_title(f"iter {i * iter_multiplier}")

    filename = os.path.join(temp_dir, f'iter_{i}.png')
    plt.savefig(filename, dpi=dpi)
    plt.close()
    return filename


# Not saving images in disk
def save_frame_3d_in_memory(frame, colors, i, dpi=80, iter_multiplier=1000,
                            xlim=(-5, 5), ylim=(-5, 5), zlim=(-5, 5),
                            marker_size=100):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    # for j in range(frame.shape[0]):
    #     ax.scatter(frame[j, 0], frame[j, 1], frame[j, 2], c=colors[j], s=marker_size)

    ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], c=colors, s=marker_size)

    margin = 0.5  # 额外空出一点边缘

    x_min, x_max = frame[:, 0].min(), frame[:, 0].max()
    y_min, y_max = frame[:, 1].min(), frame[:, 1].max()
    z_min, z_max = frame[:, 2].min(), frame[:, 2].max()

    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def create_scatter_gif_3d(ndarray, colors, gif_path='./results/output.gif', dpi=80, duration=0.5, n_jobs=-1, iter_multiplier=1000,
                          xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), marker_size=1000):
    '''
    Create a GIF from a 3D scatter plot animation.

    Parameters:
        ndarray (np.ndarray): The input data with shape (l, t, 3).
        colors (list): List of colors for each point.
        gif_path (str): Path to save the output GIF.
        dpi (int): Resolution of the output images.
        duration (float): Duration for each frame in the GIF.
        n_jobs (int): Number of parallel jobs for frame generation.
        iter_multiplier (int): Multiplier for iteration count in titles.
        xlim (tuple): Limits for the x-axis.
        ylim (tuple): Limits for the y-axis.
        zlim (tuple): Limits for the z-axis.
    '''
    # Ensure the ndarray has the correct shape for 3D data
    assert ndarray.shape[2] == 3, "ndarray must have shape (l, t, 3) for 3D data"

    temp_dir = './results/'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # print("colors:", len(colors))#9
    frames = []
    for i, frame in tqdm(enumerate(ndarray), total=ndarray.shape[0], desc='Creating GIF frames'):
        # print(f"frame {i} shape:", frame)
        # print("frame.shape:", frame.shape)#(9, 3)
        # print("x range:", frame[:, 0].min(), frame[:, 0].max())
        # print("y range:", frame[:, 1].min(), frame[:, 1].max())
        # print("z range:", frame[:, 2].min(), frame[:, 2].max())
        frames.append(save_frame_3d_in_memory(frame, colors, i, dpi, iter_multiplier, xlim, ylim, zlim, marker_size))

    # frames = [save_frame_3d_in_memory(frame, colors, i, dpi, iter_multiplier, xlim, ylim, zlim, marker_size)
    #           for i, frame in tqdm(enumerate(ndarray), total=ndarray.shape[0], desc='Creating GIF frames')]

    frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
    # # Create frames in parallel
    # filenames = Parallel(n_jobs=n_jobs)(
    #     delayed(save_frame_3d_in_memory)(frame, colors, temp_dir, i, dpi, iter_multiplier, xlim, ylim, zlim, marker_size)
    #     for i, frame in tqdm(enumerate(ndarray), total=ndarray.shape[0], desc='Creating frames')
    # )
    #
    # # Create GIF
    # frames = [Image.open(filename) for filename in filenames]
    # frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
    # for filename in filenames:
    #     os.remove(filename)


from matplotlib.animation import FuncAnimation, PillowWriter

def animate_3d_coords(coords, colors, gif_path='./results/output.gif',
                               stride=1, interval=20, marker_size=80,
                               margin=0.5, min_span=1.0, equal_aspect=True):
    """
    coords: (T,N,3) 或 (N,3)
    colors: 长度 N 的颜色
    每一帧都会按该帧的 min/max 自动设定 x/y/z 轴范围，并留出 margin。
    """
    import numpy as np
    import matplotlib.pyplot as plt

    X = np.asarray(coords, dtype=float)
    if X.ndim == 2 and X.shape[1] >= 3:
        X = X[None, :, :3]     # (N,3) -> (1,N,3)
    elif X.ndim == 3:
        if X.shape[2] < 3:
            pad = 3 - X.shape[2]
            X = np.pad(X, ((0,0),(0,0),(0,pad)), mode='constant')
        X = X[..., :3]
    else:
        raise ValueError(f"coords must be (T,N,3) or (N,3), got {X.shape}")

    X = X[::max(1, int(stride))]
    T, N, _ = X.shape

    # 颜色数量对齐
    colors = list(colors)
    if len(colors) < N:
        colors += [colors[-1]] * (N - len(colors))
    elif len(colors) > N:
        colors = colors[:N]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    if equal_aspect and hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1, 1, 1))

    scat = ax.scatter(X[0, :, 0], X[0, :, 1], X[0, :, 2], c=colors, s=marker_size)
    title = ax.set_title("Frame 0")

    def _ensure_span(lo, hi):
        span = hi - lo
        if span < min_span:
            mid = (lo + hi) / 2.0
            lo, hi = mid - min_span / 2.0, mid + min_span / 2.0
        return lo, hi

    def _set_limits_for_frame(P):  # P: (N,3)
        x_min, x_max = np.min(P[:, 0]), np.max(P[:, 0])
        y_min, y_max = np.min(P[:, 1]), np.max(P[:, 1])
        z_min, z_max = np.min(P[:, 2]), np.max(P[:, 2])

        x_min, x_max = _ensure_span(x_min, x_max)
        y_min, y_max = _ensure_span(y_min, y_max)
        z_min, z_max = _ensure_span(z_min, z_max)

        if equal_aspect and hasattr(ax, "set_box_aspect"):
            cx, cy, cz = (x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2
            r = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0 + margin
            ax.set_xlim(cx - r, cx + r)
            ax.set_ylim(cy - r, cy + r)
            ax.set_zlim(cz - r, cz + r)
        else:
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_zlim(z_min - margin, z_max + margin)

    _set_limits_for_frame(X[0])

    def update(i):
        Pi = X[i]
        scat._offsets3d = (Pi[:, 0], Pi[:, 1], Pi[:, 2])
        _set_limits_for_frame(Pi)          # 每帧自适应缩放
        title.set_text(f"Frame {i}")
        return scat, title

    anim = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)

    if gif_path:
        writer = PillowWriter(fps=max(1, int(1000/interval)))
        anim.save(gif_path, writer=writer, dpi=100)
        plt.close(fig)
    else:
        plt.show()
    return anim



def animate_2d_coords_stride(coords, colors, gif_path='./results/output.gif', interval=20, marker_size=50, stride=100):
    """
    coords: (T, N, 2) numpy array
    colors: list of N 个节点的颜色
    """
    assert coords.ndim == 3 and coords.shape[2] == 2
    T, N, _ = coords.shape
    assert len(colors) == N

    sampled_coords = coords[::stride]  # shape = (T//stride, N, 2)
    T_sampled = sampled_coords.shape[0]

    # 计算最大坐标范围以对称设置坐标轴（让原点居中）
    margin = 0.5
    x_max = np.abs(sampled_coords[:, :, 0]).max()
    y_max = np.abs(sampled_coords[:, :, 1]).max()
    R = max(x_max, y_max) + margin

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter(sampled_coords[0, :, 0], sampled_coords[0, :, 1], c=colors, s=marker_size)

    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect('equal')  # 保持长宽比一致
    ax.grid(True)

    def update(frame_idx):
        pos = sampled_coords[frame_idx]
        scat.set_offsets(pos)  # 2D only: (N, 2)
        ax.set_title(f"Frame {frame_idx * stride}", fontsize=12)
        return scat,

    ani = FuncAnimation(fig, update, frames=T_sampled, interval=interval, blit=False)
    ani.save(gif_path, dpi=80, writer=PillowWriter(fps=30))
    plt.close()


def animate_2d_coords_center_dynamic(coords, colors, center_plus, center_minus,
                                     gif_path='./results/animate_2d_coords_center.gif',
                                     interval=20, marker_size=50, stride=100):
    coords_s = coords[::stride]
    cp_s = center_plus[::stride].squeeze()
    cm_s = center_minus[::stride].squeeze()
    T_s = coords_s.shape[0]

    fig, ax = plt.subplots(figsize=(6,6))
    scat = ax.scatter(coords_s[0, :, 0], coords_s[0, :, 1], c=colors, s=marker_size)
    plus_plot, = ax.plot(cp_s[0, 0], cp_s[0, 1], 'x', color='blue', markersize=10, mew=2)
    minus_plot, = ax.plot(cm_s[0, 0], cm_s[0, 1], 'x', color='red', markersize=10, mew=2)
    ax.set_aspect('equal')
    ax.grid(True)

    def update(i):
        pts = coords_s[i]
        scat.set_offsets(pts)
        plus_plot.set_data(cp_s[i,0], cp_s[i,1])
        minus_plot.set_data(cm_s[i,0], cm_s[i,1])

        # 每帧重新计算边界
        min_x, max_x = pts[:,0].min(), pts[:,0].max()
        min_y, max_y = pts[:,1].min(), pts[:,1].max()
        xm = (max_x - min_x) * 0.05
        ym = (max_y - min_y) * 0.05
        ax.set_xlim(min_x - xm, max_x + xm)
        ax.set_ylim(min_y - ym, max_y + ym)

        ax.set_title(f'Frame {i*stride}')
        return scat, plus_plot, minus_plot

    ani = FuncAnimation(fig, update, frames=T_s, interval=interval, blit=False)
    ani.save(gif_path, writer=PillowWriter(fps=30), dpi=80)
    plt.close()


def animate_2d_coords_center(coords, colors, center_plus, center_minus, gif_path='./results/animate_2d_coords_center.gif', interval=20, marker_size=50, stride=1):
    """
    coords: (T, N, 2) numpy array
    colors: list of N 个节点的颜色
    """
    assert coords.ndim == 3 and coords.shape[2] == 2
    T, N, _ = coords.shape
    assert len(colors) == N

    # 只对帧做下采样，加快绘制
    coords_s = coords[::stride]
    cp_s = center_plus[::stride].squeeze()
    cm_s = center_minus[::stride].squeeze()
    T_s = coords_s.shape[0]

    # —— 1. 计算全局数据范围（用原始 coords 更准确） ——
    all_pts = coords.reshape(-1, 2)
    min_x, max_x = all_pts[:,0].min(), all_pts[:,0].max()
    min_y, max_y = all_pts[:,1].min(), all_pts[:,1].max()
    # 留一个 5% 的边距
    x_margin = (max_x - min_x) * 0.05
    y_margin = (max_y - min_y) * 0.05

    fig, ax = plt.subplots(figsize=(6,6))
    scat = ax.scatter(coords_s[0,:,0], coords_s[0,:,1], c=colors, s=marker_size)
    plus_plot, = ax.plot(cp_s[0,0], cp_s[0,1], 'x', color='blue', markersize=10, mew=2)
    minus_plot, = ax.plot(cm_s[0,0], cm_s[0,1], 'x', color='red', markersize=10, mew=2)

    # 设置“紧贴数据”的边界
    ax.set_xlim(min_x - x_margin, max_x + x_margin)
    ax.set_ylim(min_y - y_margin, max_y + y_margin)
    ax.set_aspect('equal')
    ax.grid(True)

    def update(i):
        scat.set_offsets(coords_s[i])
        plus_plot.set_data(cp_s[i,0], cp_s[i,1])
        minus_plot.set_data(cm_s[i,0], cm_s[i,1])
        ax.set_title(f'Frame {i*stride}')
        return scat, plus_plot, minus_plot

    ani = FuncAnimation(fig, update, frames=T_s, interval=interval, blit=False)
    ani.save(gif_path, writer=PillowWriter(fps=30), dpi=80)
    plt.close()

def animate_2d_coords_center2(coords, center_plus, center_minus, gif_path='../results/animate_2d_coords_center.gif', interval=20, marker_size=50, stride=100):
    """
    coords: (T, N, 2) numpy array
    colors: list of N 个节点的颜色
    """
    assert coords.ndim == 3 and coords.shape[2] == 2
    T, N, _ = coords.shape

    coords = coords[::stride]
    center_plus = center_plus[::stride].squeeze()
    center_minus = center_minus[::stride].squeeze()
    T_sampled = coords.shape[0]

    # 计算最大坐标范围以对称设置坐标轴（让原点居中）
    margin = 0.5
    max_range = np.max(np.abs(np.concatenate([
        coords.reshape(-1, 2),
        center_plus.reshape(-1, 2),
        center_minus.reshape(-1, 2)
    ])))
    R = max_range + margin

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter(coords[0, :, 0], coords[0, :, 1], c='green', s=marker_size)

    # Center
    plus_plot, = ax.plot(center_plus[0][0], center_plus[0][1], marker='x', color='blue', markersize=10, mew=2)
    minus_plot, = ax.plot(center_minus[0][0], center_minus[0][1], marker='x', color='red', markersize=10, mew=2)

    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect('equal')  # 保持长宽比一致
    ax.grid(True)

    def update(frame_idx):
        pos = coords[frame_idx]
        scat.set_offsets(pos)  # 2D only: (N, 2)

        # Update plus center
        plus_plot.set_data(center_plus[frame_idx][0], center_plus[frame_idx][1])
        # Update minus center
        minus_plot.set_data(center_minus[frame_idx][0], center_minus[frame_idx][1])

        ax.set_title(f"Frame {frame_idx * stride}", fontsize=12)
        return scat,

    ani = FuncAnimation(fig, update, frames=T_sampled, interval=interval, blit=False)
    ani.save(gif_path, dpi=80, writer=PillowWriter(fps=30))
    plt.close()

def animate_2d_coords_center_letter(coords, colors,
                             center_plus, center_minus,
                             labels,                          # ★ 新增
                             gif_path='./results/animate.gif',
                             interval=20, marker_size=50, stride=100):

    # -------- 参数检查 --------
    assert coords.ndim == 3 and coords.shape[2] == 2
    T, N, _ = coords.shape
    assert len(colors) == N and len(labels) == N

    # -------- 下采样 ----------
    coords       = coords[::stride]
    center_plus  = center_plus [::stride].squeeze()
    center_minus = center_minus[::stride].squeeze()
    T_sampled    = coords.shape[0]

    # -------- 坐标范围 --------
    margin = .5
    R = np.abs(np.concatenate([coords.reshape(-1, 2),
                               center_plus, center_minus])).max() + margin

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter(coords[0, :, 0], coords[0, :, 1],
                      c=colors, s=marker_size, zorder=2)

    # ★★ 给每个节点加文字
    texts = [ax.text(coords[0, i, 0], coords[0, i, 1],
                     labels[i], fontsize=11, ha='center', va='center',
                     color='black', zorder=3)
             for i in range(N)]

    plus_plot,  = ax.plot(center_plus [0, 0], center_plus [0, 1],
                          marker='x', color='blue',  mew=2, markersize=10)
    minus_plot, = ax.plot(center_minus[0, 0], center_minus[0, 1],
                          marker='x', color='red',   mew=2, markersize=10)

    ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    ax.set_aspect('equal'); ax.grid(True)

    # -------- 更新函数 --------
    def update(frame_idx):
        pos = coords[frame_idx]
        scat.set_offsets(pos)

        # ★★ 更新文字位置
        for i, txt in enumerate(texts):
            txt.set_position((pos[i, 0], pos[i, 1]))

        plus_plot .set_data(center_plus [frame_idx])
        minus_plot.set_data(center_minus[frame_idx])
        ax.set_title(f"Frame {frame_idx * stride}", fontsize=12)
        return scat, *texts, plus_plot, minus_plot   # 返回所有 artist

    ani = FuncAnimation(fig, update, frames=T_sampled,
                        interval=interval, blit=False)

    ani.save(gif_path, dpi=80, writer=PillowWriter(fps=30))
    plt.close()
