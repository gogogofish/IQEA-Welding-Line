import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

"""
正向旋转概率 p_forward 参数敏感性实验 
目标：最小化生产节拍(CT)、最小化工位负载标准差(LoadSTD)、最小化质量损失(QLoss)
敏感性实验输出：
1) 多样性：Hypervolume (HV) 箱线图（越大越好）
2) 收敛性：best_ct / best_load_std / best_qloss 的“平均收敛曲线”（越小越好）

"""

# ===================== 基础参数 =====================
NUM_TASKS = 24
NUM_STATIONS = 6
TASK_TIMES = [
    55, 65, 55, 45, 45, 35, 55, 160, 35, 70,
    100, 80, 35, 60, 60, 160, 35, 70, 450, 40,
    35, 420, 40, 35
]

ALLOWED_TOOLS = []
for task_idx in range(NUM_TASKS):
    task_id = task_idx + 1
    if task_id in {1, 4, 5, 8, 9, 13, 17, 24}:
        ALLOWED_TOOLS.append({0})
    elif task_id in {2, 3, 6, 10}:
        ALLOWED_TOOLS.append({0, 1})
    else:
        ALLOWED_TOOLS.append({2})

# 注意：为了让“任务-工具”编码维度稳定，NUM_TOOL_TYPES 应固定为 3（而不是随机初始工具集合的去重）
NUM_TOOL_TYPES = 3

TOOL_SWITCH_COST_MATRIX = np.array(
    [
        [0.0, 1.5, 1.8],
        [1.5, 0.0, 1.2],
        [1.8, 1.2, 0.0],
    ],
    dtype=float,
)

PRECEDENCE_CONSTRAINTS = [
    (0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (9, 11), (10, 11), (11, 12), (13, 15), (14, 15), (15, 16),
    (17, 18), (18, 19), (8, 20), (12, 20), (16, 20), (19, 20),
    (20, 21), (21, 22), (22, 23)
]

EPS = 1e-12

# ε-支配参数
EPS_CT = 0.5
EPS_STD = 0.25
MAX_ARCHIVE_SIZE = 20

# 质量损失参数
QUAL_ALPHA = 0.2
QUAL_BETA = 0.5
QUAL_GAMMA = 0.3
TARGET_EXPANSION = 0.025
POSITIVE_WEIGHT = 1.5
NEGATIVE_WEIGHT = 0.8


# ===================== 1) 质量模型 =====================
def simple_quality_model(task: int, station: int, tool: int):
    TOOL_BASE = {0: (0.8, 0.06, 0.02), 1: (0.6, 0.08, -0.01), 2: (0.5, 0.10, 0.03)}
    rb, db, eb = TOOL_BASE.get(tool, (0.7, 0.08, 0.0))

    task_scale = 1.0 + (TASK_TIMES[task] / (max(TASK_TIMES) + EPS)) * 0.2
    station_adjust = 1.0 + ((station % 3) - 1) * 0.2

    roughness = rb * task_scale * station_adjust
    defect_rate = min(max(db * task_scale * station_adjust, 0.0), 0.5)
    expansion = eb * task_scale * station_adjust
    return roughness, defect_rate, expansion


def calculate_quality_loss(solution) -> float:
    station_assignment, station_sequences, tool_assignment = solution
    total_roughness = total_defect_rate = total_expansion = 0.0

    for task in range(NUM_TASKS):
        station = int(station_assignment[task])
        tool = int(tool_assignment[task])
        r, d, e = simple_quality_model(task, station, tool)
        total_roughness += r
        total_defect_rate += d
        total_expansion += e

    avg_roughness = total_roughness / NUM_TASKS if NUM_TASKS > 0 else 0.0
    avg_defect_rate = total_defect_rate / NUM_TASKS if NUM_TASKS > 0 else 0.0
    avg_expansion = total_expansion / NUM_TASKS if NUM_TASKS > 0 else 0.0

    roughness_loss = avg_roughness
    defect_loss = avg_defect_rate
    expansion_diff = avg_expansion - TARGET_EXPANSION
    expansion_loss = expansion_diff * POSITIVE_WEIGHT if expansion_diff > 0 else abs(expansion_diff) * NEGATIVE_WEIGHT

    return QUAL_ALPHA * roughness_loss + QUAL_BETA * defect_loss + QUAL_GAMMA * expansion_loss


# ===================== 2) 三目标评估（含惩罚） =====================
def evaluate_welding_objectives_with_penalty(solution):
    station_assignment, station_sequences, tool_assignment = solution
    penalty = 0.0

    # 优先约束惩罚
    for pre, post in PRECEDENCE_CONSTRAINTS:
        if station_assignment[pre] > station_assignment[post]:
            penalty += 15.0
        elif station_assignment[pre] == station_assignment[post]:
            seq = station_sequences[station_assignment[pre]]
            try:
                if seq.index(pre) > seq.index(post):
                    penalty += 10.0
            except ValueError:
                penalty += 5.0

    # 工位负载（含工具切换）
    station_times = np.zeros(NUM_STATIONS)
    for s in range(NUM_STATIONS):
        seq = station_sequences[s]
        if not seq:
            continue
        station_times[s] += TASK_TIMES[seq[0]]
        current_tool = tool_assignment[seq[0]]
        for idx in range(1, len(seq)):
            task = seq[idx]
            prev_tool = current_tool
            current_tool = tool_assignment[task]
            station_times[s] += TASK_TIMES[task]
            if prev_tool != current_tool:
                station_times[s] += TOOL_SWITCH_COST_MATRIX[prev_tool, current_tool]

    ct = float(np.max(station_times))
    valid_station_times = station_times[station_times > 0]
    load_std = float(np.std(valid_station_times)) if len(valid_station_times) > 1 else 0.0
    qloss = float(calculate_quality_loss(solution))

    penalized_ct = ct + penalty
    penalized_load_std = load_std + penalty
    penalized_qloss = qloss + penalty

    return (ct, load_std, qloss), (penalized_ct, penalized_load_std, penalized_qloss), penalty


# ===================== 3) ε-支配 / 档案更新 =====================
def dominates(a, b):
    ct_dominated = a[0] <= b[0] + EPS_CT
    std_dominated = a[1] <= b[1] + EPS_STD
    qloss_dominated = a[2] < b[2]  # 质量损失严格更小

    all_dominated = ct_dominated and std_dominated and qloss_dominated

    ct_strict = a[0] < b[0] - EPS_CT
    std_strict = a[1] < b[1] - EPS_STD
    qloss_strict = a[2] < b[2]
    any_strict = ct_strict or std_strict or qloss_strict

    return all_dominated and any_strict


def calculate_crowding_distance(pareto_front):
    n = len(pareto_front)
    if n == 0:
        return []

    objectives = np.array([obj for _, obj in pareto_front], dtype=float)
    crowding_distances = np.zeros(n)

    for m in range(3):
        sorted_indices = np.argsort(objectives[:, m])
        crowding_distances[sorted_indices[0]] = np.inf
        crowding_distances[sorted_indices[-1]] = np.inf

        if n > 2:
            min_val = objectives[sorted_indices[0], m]
            max_val = objectives[sorted_indices[-1], m]
            range_val = max_val - min_val
            if range_val > EPS:
                for i in range(1, n - 1):
                    prev_obj = objectives[sorted_indices[i - 1], m]
                    next_obj = objectives[sorted_indices[i + 1], m]
                    crowding_distances[sorted_indices[i]] += (next_obj - prev_obj) / range_val

    return crowding_distances.tolist()


def update_archive(archive, cand_solution, cand_obj):
    for _, obj in archive:
        if dominates(obj, cand_obj):
            return archive

    new_archive = [(sol, obj) for sol, obj in archive if not dominates(cand_obj, obj)]
    new_archive.append((cand_solution, cand_obj))

    if len(new_archive) > MAX_ARCHIVE_SIZE:
        crowding_distances = calculate_crowding_distance(new_archive)
        sorted_indices = np.argsort(crowding_distances)[::-1]
        new_archive = [new_archive[i] for i in sorted_indices[:MAX_ARCHIVE_SIZE]]

    return new_archive


def has_empty_station(solution) -> bool:
    _, station_sequences, _ = solution
    return any(len(seq) == 0 for seq in station_sequences)


# ===================== 4) 量子观测 =====================
def observe_individual(Q_j, n_obs):
    candidates = []

    for _ in range(n_obs):
        observed_bits = []
        ptr = 0
        station_assignment = np.zeros(NUM_TASKS, dtype=int)
        tool_assignment = np.zeros(NUM_TASKS, dtype=int)
        rand_keys = np.zeros(NUM_TASKS)

        # 任务-工位
        for task in range(NUM_TASKS):
            st_probs = []
            for s in range(NUM_STATIONS):
                alpha, beta = Q_j[ptr]
                p0 = float(np.abs(alpha) ** 2)
                st_probs.append(p0)
                observed_bits.append(0 if np.random.rand() < p0 else 1)
                ptr += 1
            st_probs = np.array(st_probs, dtype=float)
            st_probs = st_probs / st_probs.sum() if st_probs.sum() > EPS else (np.ones(NUM_STATIONS) / NUM_STATIONS)
            station_assignment[task] = np.random.choice(NUM_STATIONS, p=st_probs)

        # 任务-工具（固定 3 类工具）
        for task in range(NUM_TASKS):
            tl_probs = []
            for tt in range(NUM_TOOL_TYPES):
                alpha, beta = Q_j[ptr]
                p0 = float(np.abs(alpha) ** 2)
                tl_probs.append(p0)
                observed_bits.append(0 if np.random.rand() < p0 else 1)
                ptr += 1
            tl_probs = np.array(tl_probs, dtype=float)

            if tl_probs.sum() < EPS:
                chosen_tool = np.random.choice(list(ALLOWED_TOOLS[task]))
            else:
                tl_probs /= tl_probs.sum()
                chosen_tool = int(np.random.choice(NUM_TOOL_TYPES, p=tl_probs))
                if chosen_tool not in ALLOWED_TOOLS[task]:
                    allowed = sorted(ALLOWED_TOOLS[task])
                    chosen_tool = min(allowed, key=lambda x: abs(x - chosen_tool))

            tool_assignment[task] = chosen_tool

        # 工位内排序（random keys）
        for task in range(NUM_TASKS):
            alpha, beta = Q_j[ptr]
            observed_bits.append(0 if np.random.rand() < abs(alpha) ** 2 else 1)
            rand_keys[task] = np.random.rand()
            ptr += 1

        station_sequences = [[] for _ in range(NUM_STATIONS)]
        for task, s in enumerate(station_assignment):
            station_sequences[s].append((task, rand_keys[task]))
        for s in range(NUM_STATIONS):
            station_sequences[s].sort(key=lambda x: x[1])
            station_sequences[s] = [t for t, _ in station_sequences[s]]

        sol = (station_assignment, station_sequences, tool_assignment)
        original_obj, penalized_obj, _ = evaluate_welding_objectives_with_penalty(sol)
        candidates.append((sol, np.array(observed_bits, dtype=int), original_obj, penalized_obj))

    dominated_counts = []
    for i, (_, _, _, penalized_obj_i) in enumerate(candidates):
        cnt = 0
        for j, (_, _, _, penalized_obj_j) in enumerate(candidates):
            if i != j and dominates(penalized_obj_j, penalized_obj_i):
                cnt += 1
        dominated_counts.append(cnt)

    best_index = int(np.argmin(dominated_counts))
    return candidates[best_index][0], candidates[best_index][1], candidates[best_index][2]


# ===================== 5) 参考解选择 =====================
def select_representative_solutions(pareto_archive):
    reps = []
    if not pareto_archive:
        return reps

    objs = np.array([obj for _, obj in pareto_archive], dtype=float)
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    ranges = np.maximum(maxs - mins, EPS)
    norm_objs = (objs - mins) / ranges

    utopia = np.zeros(3)
    d2utopia = np.linalg.norm(norm_objs - utopia, axis=1)
    idx_ideal = int(np.argmin(d2utopia))
    reps.append(("Ideal-point", pareto_archive[idx_ideal]))

    nadir = np.ones(3)
    line_dir = (nadir - utopia) / np.linalg.norm(nadir - utopia)
    projs = (norm_objs @ line_dir.reshape(-1, 1)) * line_dir.reshape(1, -1)
    perp_vecs = norm_objs - projs
    dist_line = np.linalg.norm(perp_vecs, axis=1)
    idx_knee = int(np.argmax(dist_line))
    reps.append(("Knee-point", pareto_archive[idx_knee]))

    return reps


# ===================== 6) 量子更新 =====================
def update_Q(Q, guided_archive, X_obs, t, max_iter, p_forward: float):
   
    pop_size, num_qubits, _ = Q.shape
    new_Q = np.copy(Q)

    valid_size = len(guided_archive)
    if valid_size == 0 or X_obs.shape[0] == 0:
        return new_Q

    valid_size = min(valid_size, X_obs.shape[0], pop_size)

    progress = t / max_iter

    reps = select_representative_solutions(guided_archive)
    if reps:
        p_random = 0.35 * (1 - progress) + 0.05
        p_knee = 0.15 + 0.15 * (1 - progress)
        p_ideal = max(1.0 - p_random - p_knee, 0.0)
        choice = np.random.choice(["ideal", "knee", "random"], p=[p_ideal, p_knee, p_random])

        if choice == "ideal" or len(reps) == 1:
            ref_solution = reps[0][1][0]
        elif choice == "knee":
            ref_solution = reps[1][1][0]
        else:
            ref_solution = random.choice(guided_archive)[0]
    else:
        ref_station = np.array([i % NUM_STATIONS for i in range(NUM_TASKS)])
        ref_tool = np.array([min(ALLOWED_TOOLS[i]) for i in range(NUM_TASKS)])
        ref_seq = [[] for _ in range(NUM_STATIONS)]
        for i, s in enumerate(ref_station):
            ref_seq[s].append(i)
        ref_solution = (ref_station, ref_seq, ref_tool)

    ref_bits = []
    station_assignment, station_sequences, tool_assignment = ref_solution

    for task in range(NUM_TASKS):
        cs = station_assignment[task]
        for s in range(NUM_STATIONS):
            ref_bits.append(1 if s == cs else 0)

    for task in range(NUM_TASKS):
        ct = tool_assignment[task]
        for tt in range(NUM_TOOL_TYPES):
            ref_bits.append(1 if tt == ct else 0)

    for task in range(NUM_TASKS):
        s = station_assignment[task]
        seq = station_sequences[s]
        if task in seq and len(seq) > 0:
            rk = seq.index(task)
            ref_bits.append(1 if rk >= len(seq) / 2 else 0)
        else:
            ref_bits.append(0)

    ref_bits = np.array(ref_bits, dtype=int)

    base_angle = 0.03 * np.pi * np.exp(-t / (max_iter / 2))
    p_mut = 0.03 * (1 - progress) + 0.01  # 简化版：保持你原来风格

    update_indices = np.random.choice(pop_size, size=valid_size, replace=False)

    for idx in range(valid_size):
        j = update_indices[idx]
        for i in range(num_qubits):
            alpha, beta = Q[j, i]
            xi = X_obs[idx, i]
            ri = ref_bits[i]

            if xi == ri:
                direction = 1.0 if np.random.rand() < p_forward else -1.0
            else:
                direction = 1.0 if ri == 1 else -1.0

            theta = base_angle * direction
            c, s = np.cos(theta), np.sin(theta)
            new_alpha = c * alpha - s * beta
            new_beta = s * alpha + c * beta

            norm = np.hypot(new_alpha, new_beta)
            if norm < EPS:
                new_alpha, new_beta = alpha, beta
                norm = np.hypot(new_alpha, new_beta)
            new_alpha /= norm
            new_beta /= norm

            if np.random.rand() < p_mut:
                if np.random.rand() < 0.5:
                    new_alpha, new_beta = new_beta, new_alpha
                else:
                    delta = np.random.uniform(-0.05 * np.pi, 0.05 * np.pi)
                    c2, s2 = np.cos(delta), np.sin(delta)
                    new_alpha, new_beta = c2 * new_alpha - s2 * new_beta, s2 * new_alpha + c2 * new_beta
                norm = np.hypot(new_alpha, new_beta)
                if norm < EPS:
                    new_alpha, new_beta = 1 / np.sqrt(2), 1 / np.sqrt(2)
                else:
                    new_alpha /= norm
                    new_beta /= norm

            new_Q[j, i, 0] = new_alpha
            new_Q[j, i, 1] = new_beta

    return new_Q


# ===================== 7) IQEA 主循环 =====================
def quantum_evolutionary_optimization(pop_size=30, max_iter=100, n_obs_base=3, p_forward=0.7):
    num_qubits = NUM_TASKS * (NUM_STATIONS + NUM_TOOL_TYPES + 1)
    Q = np.zeros((pop_size, num_qubits, 2))
    Q[:, :, 0] = Q[:, :, 1] = 1 / np.sqrt(2)

    pareto_archive = []
    history = {"best_ct": [], "best_load_std": [], "best_qloss": [], "archive_size": []}

    for t in range(max_iter):
        solutions = []
        original_objs = []
        penalized_objs = []
        X_obs = []

        progress = t / max_iter
        n_obs = n_obs_base + int(2 * progress) if progress < 1.0 else n_obs_base + 2

        for j in range(pop_size):
            sol, obs_bits, orig_obj = observe_individual(Q[j], n_obs)
            _, penalized_obj, _ = evaluate_welding_objectives_with_penalty(sol)

            if not has_empty_station(sol):
                solutions.append(sol)
                original_objs.append(orig_obj)
                penalized_objs.append(penalized_obj)
                X_obs.append(obs_bits)

        X_obs = np.array(X_obs) if X_obs else np.empty((0, num_qubits))

        for sol, orig_obj in zip(solutions, original_objs):
            if not np.any(np.isinf(orig_obj)):
                pareto_archive = update_archive(pareto_archive, sol, orig_obj)

        if pareto_archive:
            archive_objs = np.array([obj for _, obj in pareto_archive], dtype=float)
            history["best_ct"].append(float(np.min(archive_objs[:, 0])))
            history["best_load_std"].append(float(np.min(archive_objs[:, 1])))
            history["best_qloss"].append(float(np.min(archive_objs[:, 2])))
        else:
            history["best_ct"].append(np.inf)
            history["best_load_std"].append(np.inf)
            history["best_qloss"].append(np.inf)
        history["archive_size"].append(len(pareto_archive))

        guided_archive = [(sol, pen_obj) for sol, pen_obj in zip(solutions, penalized_objs)]
        Q = update_Q(Q, guided_archive, X_obs, t, max_iter, p_forward=p_forward)

    return pareto_archive, history


# ===================== 8) HV =====================
def calculate_hypervolume(points, ref_point):
    
    if points is None or len(points) == 0:
        return 0.0

    P = np.array(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        return 0.0

    # 过滤掉非有限值
    mask = np.all(np.isfinite(P), axis=1)
    P = P[mask]
    if len(P) == 0:
        return 0.0

    P = P[np.argsort(P[:, 0])]
    hv = 0.0
    prev_x = ref_point[0]
    for x, y, z in P:
        if x < prev_x:
            dx = prev_x - x
            dy = max(ref_point[1] - y, 0.0)
            dz = max(ref_point[2] - z, 0.0)
            hv += dx * dy * dz
            prev_x = x
    return float(max(hv, 0.0))


# ===================== 9) 敏感性实验：配对种子，多次重复 =====================
def run_p_forward_sensitivity_experiment(
    p_list,
    n_runs=20,
    pop_size=30,
    max_iter=100,
    n_obs_base=3,
    seed0=10000,
    verbose=True,
):
    """配对设计：同一 seed 在不同 p_forward 下各跑一次，保证曲线对比和HV分布更公平。"""
    seeds = [seed0 + k for k in range(n_runs)]

    # 先跑全实验，缓存每次运行的Pareto点集，用于统一 ref_point
    cached_objs = {p: [] for p in p_list}
    cached_hist = {p: [] for p in p_list}

    total = len(seeds) * len(p_list)
    cur = 0

    for seed in seeds:
        for p in p_list:
            cur += 1
            if verbose:
                print(f"Progress {cur:4d}/{total} | seed={seed} | p_forward={p:.2f}", end="\r")

            np.random.seed(seed)
            random.seed(seed)

            archive, history = quantum_evolutionary_optimization(
                pop_size=pop_size, max_iter=max_iter, n_obs_base=n_obs_base, p_forward=p
            )

            cached_hist[p].append(history)

            if archive:
                objs = np.array([obj for _, obj in archive], dtype=float)
            else:
                objs = np.empty((0, 3), dtype=float)

            cached_objs[p].append(objs)

    if verbose:
        print("\nExperiment finished.")

    all_points = []
    for p in p_list:
        for objs in cached_objs[p]:
            if objs.size:
                all_points.append(objs)
    if all_points:
        all_points = np.vstack(all_points)
        ref_point = np.max(all_points, axis=0) * 1.1
    else:
        # 兜底
        ref_point = np.array([1000.0, 100.0, 10.0], dtype=float)

    results = {"hv": {p: [] for p in p_list}, "history": {p: [] for p in p_list}, "ref_point": ref_point}

    for p in p_list:
        results["history"][p] = cached_hist[p]
        for objs in cached_objs[p]:
            hv = calculate_hypervolume(objs, ref_point=ref_point)
            results["hv"][p].append(hv)

    return results


# ===================== 10) 可视化 =====================
def _avg_curve(histories, key, smooth_window=3):
    # 对齐到 max_iter，缺失/inf 忽略
    max_len = max(len(h.get(key, [])) for h in histories) if histories else 0
    if max_len == 0:
        return np.array([])

    curve = []
    for t in range(max_len):
        vals = []
        for h in histories:
            arr = h.get(key, [])
            if t < len(arr) and np.isfinite(arr[t]):
                vals.append(arr[t])
        curve.append(np.mean(vals) if vals else np.nan)

    y = np.array(curve, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return np.array([])

    if smooth_window is not None and smooth_window >= 2 and y.size > smooth_window:
        y = np.convolve(y, np.ones(smooth_window) / smooth_window, mode="valid")
    return y


def plot_sensitivity_results(results, p_list, out_png="p_forward_sensitivity.png"):
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    labels = [f"{p*100:.0f}%" for p in p_list]

    # (a) HV 箱线图
    hv_data = [results["hv"][p] for p in p_list]
    bp = axes[0, 0].boxplot(hv_data, labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    # axes[0, 0].set_title("(a) HV（-）")
    axes[0, 0].set_xlabel("p_forward")
    axes[0, 0].set_ylabel("HV(-)")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # (b) best_ct 平均曲线
    for i, p in enumerate(p_list):
        y = _avg_curve(results["history"][p], "best_ct", smooth_window=3)
        if y.size:
            axes[0, 1].plot(y, label=f"{p*100:.0f}%", linewidth=2.0, alpha=0.9, color=colors[i])
    # axes[0, 1].set_title("(b) best_ct 平均收敛曲线（越小越好）")
    axes[0, 1].set_xlabel("iteration number")
    axes[0, 1].set_ylabel("CT(s)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # (c) best_load_std 平均曲线
    for i, p in enumerate(p_list):
        y = _avg_curve(results["history"][p], "best_load_std", smooth_window=3)
        if y.size:
            axes[1, 0].plot(y, label=f"{p*100:.0f}%", linewidth=2.0, alpha=0.9, color=colors[i])
    # axes[1, 0].set_title("(c) best_load_std 平均收敛曲线（越小越好）")
    axes[1, 0].set_xlabel("iteration number")
    axes[1, 0].set_ylabel("LoadSTD(s)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # (d) best_qloss 平均曲线
    for i, p in enumerate(p_list):
        y = _avg_curve(results["history"][p], "best_qloss", smooth_window=3)
        if y.size:
            axes[1, 1].plot(y, label=f"{p*100:.0f}%", linewidth=2.0, alpha=0.9, color=colors[i])
    # axes[1, 1].set_title("(d) best_qloss 平均收敛曲线（越小越好）")
    axes[1, 1].set_xlabel("iteration number")
    axes[1, 1].set_ylabel("Qloss")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {out_png}")
    print(f"HV ref_point used (fixed for comparability): {results['ref_point']}")


def print_hv_summary(results, p_list):
    print("\n" + "=" * 70)
    print("p_forward 敏感性：HV 汇总（越大越好）")
    print("=" * 70)
    for p in p_list:
        hv = np.array(results["hv"][p], dtype=float)
        hv = hv[np.isfinite(hv)]
        mean = float(np.mean(hv)) if hv.size else 0.0
        std = float(np.std(hv)) if hv.size else 0.0
        print(f"p_forward={p:.2f} | HV = {mean:.3f} ± {std:.3f} (n={hv.size})")


# ===================== 主程序 =====================
if __name__ == "__main__":
    P_LIST = [0.60, 0.65, 0.70, 0.75]
    N_RUNS = 20
    POP_SIZE = 30
    MAX_ITER = 100
    N_OBS_BASE = 3

    print("=" * 70)
    print("Start p_forward sensitivity experiment")
    print(f"P_LIST={P_LIST} | runs={N_RUNS} | pop={POP_SIZE} | iter={MAX_ITER} | n_obs_base={N_OBS_BASE}")
    print("=" * 70)

    results = run_p_forward_sensitivity_experiment(
        p_list=P_LIST,
        n_runs=N_RUNS,
        pop_size=POP_SIZE,
        max_iter=MAX_ITER,
        n_obs_base=N_OBS_BASE,
        seed0=10000,
        verbose=True,
    )

    print_hv_summary(results, P_LIST)
    plot_sensitivity_results(results, P_LIST, out_png="p_forward_sensitivity1.0.png")
