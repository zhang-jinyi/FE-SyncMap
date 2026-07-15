import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import time
from tqdm import tqdm
_HAS_TQDM = True

def _softmax(logits, axis=-1):
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    s = e.sum(axis=axis, keepdims=True)
    return e / s

class SyncMap:
    def __init__(
        self,
        input_size,
        dimensions,
        adaptation_rate,
        space_size=15.0,
        a=0,
        pos_k=3,
        use_hier=False,
        db_eps=3,
        db_min_samples=2,
        seed=412,

            # ======== 自由能/惊讶 相关超参 ========
        c_decay=0.01,
        # ======== 可视化内存保护（快速开关） ========
        record_traj=True,  # False = 完全不存轨迹
        viz_stride=200,  # 仅在 record_traj=True 时生效

            # ======== 进度显示相关 ========
        show_progress=True,       # 是否显示进度
        use_tqdm=True,            # 优先使用 tqdm（若可用）
        progress_desc="SyncMap",  # 进度条标题
        progress_every=200,       # 无 tqdm 时的打印步频
        progress_hook=None        # 可选回调：fn(step:int,total:int,info:dict)
    ):
        self.record_traj = bool(record_traj)
        self.viz_stride = int(viz_stride)
        self.draw_3d_nodes = []
        self.rng = np.random.default_rng(seed)
        self.organized = False
        self.space_size = float(space_size)
        self.dimensions = int(dimensions)
        self.input_size = int(input_size)
        self.syncmap = self.rng.random((self.input_size, self.dimensions))
        self.adaptation_rate = float(adaptation_rate)
        self._clip_to_ball()

        # 论文→代码：记忆窗与对称激活
        self.a = float(a)
        self.pos_k = int(pos_k)

        # 聚类参数
        self.use_hier = bool(use_hier)
        self.db_eps = float(db_eps)
        self.db_min_samples = int(db_min_samples)

        # ======== 自由能模块状态 ========
        # gamma 参数保留用于兼容旧接口，但不再参与 softmax（等价于固定 gamma=1）
        self.gamma = 1.0
        self.c_decay = float(c_decay)
        self.C = np.zeros((self.input_size, self.input_size), dtype=np.float64)
        # === Cached row sums for p_row (incremental; math unchanged) ===
        self.row_sums = self.C.sum(axis=1).astype(np.float64)
        self.total_sum = float(self.row_sums.sum())


        # ======== 进度配置 ========
        self.show_progress = bool(show_progress)
        self.use_tqdm = bool(use_tqdm) and _HAS_TQDM
        self.progress_desc = str(progress_desc)
        self.progress_every = int(progress_every)
        self.progress_hook = progress_hook  # 可传 lambda step,total,info: ...
        self.draw_3d_nodes = []
        self.center_plus = []
        self.center_minus = []


    # ---------- 进度设置（可运行时修改） ----------
    def set_progress(self, show=None, desc=None, use_tqdm=None, every=None):
        if show is not None:
            self.show_progress = bool(show)
        if desc is not None:
            self.progress_desc = str(desc)
        if use_tqdm is not None:
            self.use_tqdm = bool(use_tqdm) and _HAS_TQDM
        if every is not None:
            self.progress_every = int(every)

    def _clip_to_ball(self):
        # r: (N,1)
        r = np.linalg.norm(self.syncmap, axis=1, keepdims=True)
        # over: (N,) —— 只在行维度做布尔索引
        over = (r[:, 0] > self.space_size)
        if np.any(over):
            # r[over]: (K,1) 与 (K,D) 广播相乘，形状匹配
            self.syncmap[over] *= (self.space_size /r[over])



    def _safe_dir(self, center):
        diff = (center[None, :] - self.syncmap)               # (N, D)
        den = np.linalg.norm(diff, axis=1, keepdims=True)     # (N, 1)
        return diff / den

    # --------- 自由能辅助：p_hat、惊讶与选择 ---------
    def _p_hat_column(self, r):
        """
        返回 p̂(i|r)：对“列 r”按列求和归一（无加性平滑）。
        兼容 r 为 float/0-d ndarray，并做越界检查。
        """
        r = int(np.asarray(r).reshape(()))  # ← 强制 Python int
        if r < 0 or r >= self.input_size:
            raise IndexError(f"column index r={r} out of range [0,{self.input_size})")

        col = self.C[:, [r]]  # ← 花式索引，稳定得到 (N,1)
        denom = float(col.sum())
        if denom <= 0.0:
            # 若该列还没有任何计数，返回均匀分布以避免除零
            return np.full((self.input_size, 1), 1.0 / self.input_size, dtype=float)
        eps = 1e-12  # 或者 1e-9 / 1e-6 视你想要的先验强度
        return (col + eps) / (denom + eps * self.input_size)

    def _adaptive_lr(self, vals):
        import numpy as np
        v = np.asarray(vals, dtype=float).ravel()
        if v.size == 0:
            return v
        m = float(np.mean(v))
        s = float(np.std(v))
        # 若这批分数几乎没有差异，退化为常数步长
        if not np.isfinite(s) or s < 1e-12:
            return np.full_like(v, 0.5*self.adaptation_rate, dtype=float)
        z = (v - m) / s  # z-score
        scale = np.abs(z)
        return self.adaptation_rate * scale

    def _update_transition_counts(self, r, pos_idx_all):
        if r < 0 or r >= self.input_size:
            return
        pos_idx_all = np.asarray(pos_idx_all, dtype=int)
        w_r = self.syncmap[r]
        Wpos = self.syncmap[pos_idx_all]                          # (Kp,D)
        d2   = np.sum((Wpos - w_r[None, :])**2, axis=1)           # (Kp,)
        # 直接用 -d2 做 softmax（不做缩放）
        q    = _softmax((-d2).reshape(1, -1), axis=1).ravel()  # (Kp,), ∑q=1

        # 只遗忘/更新列 r，并同步维护行和缓存（行边缘概率的增量版）
        col_old = self.C[:, r].copy()
        s_old   = float(col_old.sum())

        # 衰减列 r
        self.C[:, r] *= (1.0 - self.c_decay)
        self.row_sums -= self.c_decay * col_old

        # 加入 c_decay * q 到被选中的行（∑q=1）
        self.C[pos_idx_all, r] += self.c_decay * q
        self.row_sums[pos_idx_all] += self.c_decay * q

        # 维护 total_sum = ∑_i row_sums[i]
        self.total_sum += (- self.c_decay * s_old) + (self.c_decay * 1.0)

    def _p_row_marginal(self):
        # Use cached row_sums / total_sum (incremental; mathematically equivalent to summing C each step)
        beta = self.beta_smooth
        denom = float(self.total_sum) + beta * self.input_size  # 标量
        return (self.row_sums + beta) / denom


    def inputGeneral(self, X, r_seq=None):
        """
        X: [T, N]
        r_seq: [T,] 每一步的真实当前节点索引（trajectory）
        """
        T = X.shape[0]
        last_print = -self.progress_every
        t0 = time.time()
        if r_seq is not None:
            r_seq = np.asarray(r_seq).reshape(-1)
            if len(r_seq) != T:
                raise ValueError(f"r_seq length {len(r_seq)} != T {T}")

        iterator = range(T)
        pbar = None
        # Progress bar (tqdm) - does NOT change algorithm behavior
        if self.show_progress and self.use_tqdm:
            pbar = tqdm(range(T), total=T, desc=self.progress_desc, leave=True, dynamic_ncols=True)

        for t in (pbar if pbar is not None else iterator):
            x = X[t]
            thr = self.a
            pos_idx_all = np.asarray(np.where(x > thr)[0], dtype=int)
            neg_idx_all = np.asarray(np.where(x <= thr)[0], dtype=int)
            r = int(r_seq[t])

            # 后面保持不变
            self._update_transition_counts(r, pos_idx_all)
            p_col = self._p_hat_column(r)[:, 0]

            # ===  正样本选择：惊讶===
            k = min(self.pos_k, pos_idx_all.size)
            if pos_idx_all.size > k:
                # 1) 边缘概率 p_row 与 SPMI
                log_p_col = np.log(np.clip(p_col, 0, 1.0))  # (N,)
                # 2) 正样本惊讶
                surp_pos_all = -log_p_col[pos_idx_all]  # 惊讶
                logw_pos = np.log(surp_pos_all)

                w_pos = _softmax(logw_pos.reshape(1, -1), axis=1).ravel()  # gamma 已固定为 1
                pos_idx = self.rng.choice(pos_idx_all, size=k, replace=False, p=w_pos)
            else:
                pos_idx = pos_idx_all

            # ---  对称负采样（不依赖几何，泛化强） ---
            if neg_idx_all.size > k:
                # 预先已有：p_col = self._p_hat_column(r_idx)[:, 0]   # (N,)
                # 负例权重：高不确定（1 − p̂）
                w_prob = 1.0 - p_col[neg_idx_all]
                logw = np.log(w_prob)

                # gamma 已固定为 1
                w = _softmax(logw.reshape(1, -1), axis=1).ravel()

                neg_idx = self.rng.choice(neg_idx_all, size=k, replace=False, p=w)
            else:
                neg_idx = neg_idx_all[:k]

            # ---------- 质心 ----------
            p_col = self._p_hat_column(r)[:, 0]
            surp_pos_sel = -np.log(p_col[pos_idx])
            neg_score_sel = 1.0 - p_col[neg_idx]
            omega_pos = np.ones_like(surp_pos_sel)
            omega_neg = np.ones_like(neg_score_sel)

            alpha_pos = self._adaptive_lr(surp_pos_sel)
            alpha_neg = self._adaptive_lr(neg_score_sel)
            Wp = self.syncmap[pos_idx]
            Wn = self.syncmap[neg_idx]
            cp = (omega_pos[:, None] * Wp).sum(axis=0) / (omega_pos.sum() )
            cn = (omega_neg[:, None] * Wn).sum(axis=0) / (omega_neg.sum() )

            dir_pos_sel = cp[None, :] - Wp
            dir_neg_sel = cn[None, :] - Wn
            dir_pos_sel /= (np.linalg.norm(dir_pos_sel, axis=1, keepdims=True) + 1e-12)
            dir_neg_sel /= (np.linalg.norm(dir_neg_sel, axis=1, keepdims=True) + 1e-12)

            delta_pos = (alpha_pos[:, None] * dir_pos_sel)  # (k,D)
            delta_neg = (alpha_neg[:, None] * dir_neg_sel)  # (k,D)

            self.syncmap[pos_idx] += delta_pos
            self.syncmap[neg_idx] -= delta_neg
            self._clip_to_ball()

            if self.record_traj and (t % self.viz_stride == 0):
                self.draw_3d_nodes.append(self.syncmap[:, :3].copy())  # 或你自己的 _viz3_current()
                self.center_plus.append(cp[:2].copy())
                self.center_minus.append(cn[:2].copy())


            # ---------- 进度打印/回调 ----------
            if pbar is None and self.show_progress and (t - last_print >= self.progress_every or t == T - 1):
                pct = (t + 1) * 100.0 / T
                elapsed = time.time() - t0
                print(f"{self.progress_desc}: {t+1}/{T} ({pct:.1f}%) | elapsed {elapsed:.1f}s", end="\r")
                last_print = t
            if self.progress_hook is not None and ((t + 1) % max(1, self.progress_every) == 0 or t == T - 1):
                info = {"elapsed": time.time() - t0}
                self.progress_hook(t + 1, T, info)

        if pbar is not None:
            pbar.close()
        if self.show_progress and not self.use_tqdm:
            print()  # 换行美观

    def input(self, X, r_seq=None):
        self.inputGeneral(X, r_seq=r_seq)
        return

    def organize(self):
        self.organized = True
        data = self.syncmap

        if self.show_progress:
            print(f"[{self.progress_desc}] Clustering...", end="\r")
        if self.use_hier:
            hc = AgglomerativeClustering(
                n_clusters=None, distance_threshold=self.db_eps, linkage='ward'
            )
            labels = hc.fit_predict(data)
        else:
            labels = DBSCAN(eps=self.db_eps, min_samples=self.db_min_samples).fit_predict(data)
        self.labels = labels
        if self.show_progress:
            print(f"[{self.progress_desc}] Clustering... done.         ")
        traj = (np.asarray(self.draw_3d_nodes, dtype=np.float32)
                if self.record_traj else np.empty((0, self.input_size, 3), dtype=np.float32))
        return self.labels, traj

    def get_syncmap(self, isMovMean=False):
        if isMovMean:
            self.syncmap_movmean = np.mean(np.asarray(self.syncmap_movmean_list), axis=0)
            return self.syncmap_movmean, np.array(self.draw_3d_nodes), np.array(self.center_plus), np.array(
                self.center_minus)
        else:
            return self.syncmap, np.array(self.draw_3d_nodes), np.array(self.center_plus), np.array(self.center_minus)

