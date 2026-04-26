import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

"""焊装线三目标IQEA算法（变异模式对比实验版）
目标：最小化生产节拍(CT)、最小化工位负载标准差(LoadSTD)、最小化质量损失(QLoss)
变异模式对比：四种不同比例配置
"""

# ==== 基础参数 ====
NUM_TASKS = 24  # 焊装线工序任务数
NUM_STATIONS = 6  # 工作站数量
# 任务工时（秒）
TASK_TIMES = [
    55, 65, 55, 45, 45, 35, 55, 160, 35, 70,
    100, 80, 35, 60, 60, 160, 35, 70, 450, 40,
    35, 420, 40, 35
]

# 各任务允许使用的工具集合（0/1/2代表三种工具）
ALLOWED_TOOLS = []
for task_idx in range(NUM_TASKS):
    task_id = task_idx + 1  # 任务编号从1开始
    if task_id in {1, 4, 5, 8, 9, 13, 17, 24}:
        ALLOWED_TOOLS.append({0})
    elif task_id in {2, 3, 6, 10}:
        ALLOWED_TOOLS.append({0, 1})
    else:
        ALLOWED_TOOLS.append({2})

# 初始工具分配（随机选择允许的工具）
TOOL_TYPES = [random.choice(list(tools)) for tools in ALLOWED_TOOLS]
NUM_TOOL_TYPES = len(set(TOOL_TYPES))
# 工具切换成本矩阵（单位：秒）
TOOL_SWITCH_COST_MATRIX = np.array([
    [0.0, 1.5, 1.8],
    [1.5, 0.0, 1.2],
    [1.8, 1.2, 0.0],
], dtype=float)

# 任务优先约束关系（(前驱任务, 后继任务)）
PRECEDENCE_CONSTRAINTS = [
    (0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (9, 11), (10, 11), (11, 12), (13, 15), (14, 15), (15, 16),
    (17, 18), (18, 19), (8, 20), (12, 20), (16, 20), (19, 20),
    (20, 21), (21, 22), (22, 23)
]

EPS = 1e-12  # 数值计算精度控制

# ==== ε-支配参数 ====
EPS_CT = 0.5  # 降低生产节拍的ε容差
EPS_STD = 0.25  # 降低负载标准差的ε容差

# ==== 档案大小限制 ====
MAX_ARCHIVE_SIZE = 20

# ==== 质量相关参数 ====
QUAL_ALPHA = 0.2  # 粗糙度权重系数
QUAL_BETA = 0.5  # 缺陷率权重系数
QUAL_GAMMA = 0.3  # 膨胀量权重系数
TARGET_EXPANSION = 0.025  # 目标膨胀量（
POSITIVE_WEIGHT = 1.5  # 膨胀量正偏差权重
NEGATIVE_WEIGHT = 0.8  # 膨胀量负偏差权重

# ==== CT阈值参数 ====
MAX_CT_THRESHOLD = 600.0  # 生产节拍最大允许阈值

# ==== 变异模式配置 ====
MUTATION_MODES = {
    "mode1 (50-40-10)": (0.5, 0.4, 0.1),  # 重置50%, 翻转40%, 微调10%
    "mode2 (50-30-20)": (0.5, 0.3, 0.2),  # 重置50%, 翻转30%, 微调20%
    "mode3 (40-40-20)": (0.4, 0.4, 0.2),  # 重置40%, 翻转40%, 微调20%
    "mode4 (40-30-30)": (0.4, 0.3, 0.3)  # 重置40%, 翻转30%, 微调30%
}

# ==== 实验参数 ====
NUM_RUNS = 30  # 每种模式运行次数
POP_SIZE = 30
MAX_ITER = 100
N_OBS_BASE = 3


# ==== 质量损失计算模块 ====
def simple_quality_model(task: int, station: int, tool: int):
    """根据任务、工位、工具计算质量参数（粗糙度、缺陷率、膨胀量）"""
    TOOL_BASE = {0: (0.8, 0.06, 0.02), 1: (0.6, 0.08, -0.01), 2: (0.5, 0.10, 0.03)}
    rb, db, eb = TOOL_BASE.get(tool, (0.7, 0.08, 0.0))

    task_scale = 1.0 + (TASK_TIMES[task] / (max(TASK_TIMES) + EPS)) * 0.2
    station_adjust = 1.0 + ((station % 3) - 1) * 0.2

    roughness = rb * task_scale * station_adjust
    defect_rate = min(max(db * task_scale * station_adjust, 0.0), 0.5)
    expansion = eb * task_scale * station_adjust

    return roughness, defect_rate, expansion


def calculate_quality_loss(solution) -> float:
    """计算综合质量损失（三目标中的"质量损失"目标）"""
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


# ==== 2. 多目标评估模块 ====
def evaluate_welding_objectives_with_penalty(solution):
    """评估三目标值（含约束违反惩罚），返回(原始目标值, 惩罚后目标值, 惩罚值)"""
    station_assignment, station_sequences, tool_assignment = solution
    penalty = 0.0

    # 检查优先约束违反
    for pre, post in PRECEDENCE_CONSTRAINTS:
        if station_assignment[pre] > station_assignment[post]:
            penalty += 15.0
        elif station_assignment[pre] == station_assignment[post]:
            seq = station_sequences[station_assignment[pre]]
            try:
                if seq.index(pre) >= seq.index(post):
                    penalty += 10.0
            except ValueError:
                penalty += 0.0

    # 计算工位负载（含工具切换成本）
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

    # 计算三目标原始值
    ct = float(np.max(station_times))
    valid_station_times = station_times[station_times > 0]
    load_std = float(np.std(valid_station_times)) if len(valid_station_times) > 1 else 0.0
    qloss = float(calculate_quality_loss(solution))

    # 计算惩罚后目标值
    penalized_ct = ct + penalty
    penalized_load_std = load_std + penalty
    penalized_qloss = qloss + penalty

    return (ct, load_std, qloss), (penalized_ct, penalized_load_std, penalized_qloss), penalty


# ==== 3. 量子观测模块 ====
def observe_individual(Q_j, n_obs):
    """对单个量子个体进行n_obs次观测，选择支配次数最少的解"""
    candidates = []

    for _ in range(n_obs):
        observed_bits = []
        ptr = 0
        station_assignment = np.zeros(NUM_TASKS, dtype=int)
        tool_assignment = np.zeros(NUM_TASKS, dtype=int)
        rand_keys = np.zeros(NUM_TASKS)

        # 观测"任务-工位"分配
        for task in range(NUM_TASKS):
            st_probs = []
            for s in range(NUM_STATIONS):
                alpha, beta = Q_j[ptr]
                p0 = float(np.abs(alpha) ** 2)
                st_probs.append(p0)
                observed_bits.append(0 if np.random.rand() < p0 else 1)
                ptr += 1
            st_probs = np.array(st_probs)
            st_probs = st_probs / st_probs.sum() if st_probs.sum() > EPS else (np.ones(NUM_STATIONS) / NUM_STATIONS)
            station_assignment[task] = np.random.choice(NUM_STATIONS, p=st_probs)

        # 观测"任务-工具"分配
        for task in range(NUM_TASKS):
            tl_probs = []
            for tt in range(NUM_TOOL_TYPES):
                alpha, beta = Q_j[ptr]
                p0 = float(np.abs(alpha) ** 2)
                tl_probs.append(p0)
                observed_bits.append(0 if np.random.rand() < p0 else 1)
                ptr += 1
            tl_probs = np.array(tl_probs)
            if tl_probs.sum() < EPS:
                chosen_tool = np.random.choice(list(ALLOWED_TOOLS[task]))
            else:
                tl_probs /= tl_probs.sum()
                chosen_tool = np.random.choice(NUM_TOOL_TYPES, p=tl_probs)
                if chosen_tool not in ALLOWED_TOOLS[task]:
                    allowed = sorted(ALLOWED_TOOLS[task])
                    chosen_tool = min(allowed, key=lambda x: abs(x - chosen_tool))
            tool_assignment[task] = chosen_tool

        # 观测"工位内任务排序"
        for task in range(NUM_TASKS):
            alpha, beta = Q_j[ptr]
            observed_bits.append(0 if np.random.rand() < abs(alpha) ** 2 else 1)
            rand_keys[task] = np.random.rand()
            ptr += 1

        # 生成工位内任务序列
        station_sequences = [[] for _ in range(NUM_STATIONS)]
        for task, s in enumerate(station_assignment):
            station_sequences[s].append((task, rand_keys[task]))
        for s in range(NUM_STATIONS):
            station_sequences[s].sort(key=lambda x: x[1])
            station_sequences[s] = [t for t, _ in station_sequences[s]]

        # 评估当前观测解
        sol = (station_assignment, station_sequences, tool_assignment)
        original_obj, penalized_obj, _ = evaluate_welding_objectives_with_penalty(sol)
        candidates.append((sol, np.array(observed_bits, dtype=int), original_obj, penalized_obj))

    # 选择被支配次数最少的解
    dominated_counts = []
    for i, (_, _, _, penalized_obj_i) in enumerate(candidates):
        cnt = 0
        for j, (_, _, _, penalized_obj_j) in enumerate(candidates):
            if i != j and dominates(penalized_obj_j, penalized_obj_i):
                cnt += 1
        dominated_counts.append(cnt)
    best_index = int(np.argmin(dominated_counts))

    return candidates[best_index][0], candidates[best_index][1], candidates[best_index][2]


# ==== 4. Pareto前沿核心模块 ====
def dominates(a, b):
    """ε-支配判断：解a是否ε-支配解b（CT/STD允许容差，QLoss必须严格更小）"""
    ct_dominated = a[0] <= b[0] + EPS_CT
    std_dominated = a[1] <= b[1] + EPS_STD
    qloss_dominated = a[2] < b[2]

    all_dominated = ct_dominated and std_dominated and qloss_dominated

    ct_strict = a[0] < b[0] - EPS_CT
    std_strict = a[1] < b[1] - EPS_STD
    qloss_strict = a[2] < b[2]

    any_strict = ct_strict or std_strict or qloss_strict

    return all_dominated and any_strict


def update_archive(archive, cand_solution, cand_obj):
    """更新帕累托档案，使用ε-支配"""
    if cand_obj[0] > MAX_CT_THRESHOLD:
        return archive

    for _, obj in archive:
        if dominates(obj, cand_obj):
            return archive

    new_archive = [(sol, obj) for sol, obj in archive if not dominates(cand_obj, obj)]
    new_archive.append((cand_solution, cand_obj))

    return new_archive


def select_representative_solutions(pareto_archive):
    """从帕累托档案中选择参考解"""
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


# ==== 5. 量子进化更新模块（修改变异操作） ====
def update_Q(Q, guided_archive, X_obs, t, max_iter, mutation_mode):
    """更新量子种群的量子态（按照指定变异模式比例）"""
    pop_size, num_qubits, _ = Q.shape
    new_Q = np.copy(Q)
    valid_size = len(guided_archive)
    if valid_size == 0:
        return new_Q
    progress = t / max_iter

    # 选择参考解
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

    # 编码参考解为比特串
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
    ref_bits = np.array(ref_bits, int)

    # 旋转门与变异参数
    base_angle = 0.03 * np.pi * np.exp(-t / (max_iter / 2))
    archive_size = len(guided_archive)
    p_mut_min, p_mut_max = 0.006, 0.06
    size_factor = 1.0 / (1.0 + np.log1p(max(archive_size, 1)))
    p_mut = p_mut_min + (p_mut_max - p_mut_min) * (1 - progress) * size_factor

    # 根据变异模式设置比例
    p_reinit_ratio, p_flip_ratio, p_rotate_ratio = MUTATION_MODES[mutation_mode]
    p_reinit = p_reinit_ratio * p_mut
    p_flip = p_flip_ratio * p_mut

    phi_max = (0.08 * np.pi) * (1 - progress) + 0.02 * np.pi

    # 仅对有效解对应的量子个体更新
    update_indices = np.random.choice(pop_size, size=valid_size, replace=False)
    for idx in range(valid_size):
        j = update_indices[idx]
        for i in range(num_qubits):
            alpha, beta = Q[j, i]
            xi = X_obs[idx, i]
            ri = ref_bits[i]

            # 旋转门更新
            if xi == ri:
                direction = 1.0 if np.random.rand() < 0.7 else -1.0
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

            # 按照指定比例进行变异操作
            p1 = float(abs(new_beta) ** 2)
            polarization = abs(p1 - 0.5)
            boost = 1.0 + 1.5 * max(0.0, polarization - 0.35)
            if np.random.rand() < p_mut * boost:
                r = np.random.rand()
                if r < p_reinit_ratio:
                    # 模式1：重置为均匀叠加态
                    new_alpha = new_beta = 1 / np.sqrt(2)
                elif r < p_reinit_ratio + p_flip_ratio:
                    # 模式2：概率幅翻转 + 微小扰动
                    new_alpha, new_beta = new_beta, new_alpha
                    jitter = np.random.uniform(-0.01 * np.pi, 0.01 * np.pi)
                    c2, s2 = np.cos(jitter), np.sin(jitter)
                    new_alpha, new_beta = c2 * new_alpha - s2 * new_beta, s2 * new_alpha + c2 * new_beta
                else:
                    # 模式3：旋转门微调
                    delta = np.random.uniform(-phi_max, phi_max)
                    c2, s2 = np.cos(delta), np.sin(delta)
                    new_alpha, new_beta = c2 * new_alpha - s2 * new_beta, s2 * new_alpha + c2 * new_beta

                # 归一化
                norm = np.hypot(new_alpha, new_beta)
                if norm < EPS:
                    new_alpha, new_beta = 1 / np.sqrt(2), 1 / np.sqrt(2)
                else:
                    new_alpha /= norm
                    new_beta /= norm

            new_Q[j, i, 0] = new_alpha
            new_Q[j, i, 1] = new_beta

    return new_Q


# ==== 6. QLoss局部改进模块 ====
def local_qloss_improvement(solution, obj_values):
    """在不恶化CT和LoadSTD的前提下，通过微调工具分配来改进QLoss"""
    station_assignment, station_sequences, tool_assignment = solution
    ct, load_std, qloss = obj_values

    improved = False
    new_tool_assignment = tool_assignment.copy()

    for task in range(NUM_TASKS):
        original_tool = tool_assignment[task]
        allowed_tools = list(ALLOWED_TOOLS[task])

        if len(allowed_tools) <= 1:
            continue

        for new_tool in allowed_tools:
            if new_tool == original_tool:
                continue

            test_tool_assignment = new_tool_assignment.copy()
            test_tool_assignment[task] = new_tool
            test_solution = (station_assignment, station_sequences, test_tool_assignment)
            test_obj, _, _ = evaluate_welding_objectives_with_penalty(test_solution)
            test_ct, test_std, test_qloss = test_obj

            ct_ok = test_ct <= ct
            std_ok = test_std <= load_std
            qloss_improved = test_qloss < qloss - EPS

            if ct_ok and std_ok and qloss_improved:
                new_tool_assignment = test_tool_assignment
                ct, load_std, qloss = test_ct, test_std, test_qloss
                improved = True
                break

    if improved:
        improved_solution = (station_assignment, station_sequences, new_tool_assignment)
        improved_obj = (ct, load_std, qloss)
        return improved_solution, improved_obj

    return solution, obj_values


# ==== 7. IQEA主循环（支持变异模式参数） ====
def quantum_evolutionary_optimization(pop_size=30, max_iter=100, n_obs_base=3, mutation_mode="mode 1 (50-40-10)"):
    """三目标量子进化算法主循环，返回(帕累托档案, 优化历史记录)"""
    num_qubits = NUM_TASKS * (NUM_STATIONS + NUM_TOOL_TYPES + 1)
    Q = np.zeros((pop_size, num_qubits, 2))
    Q[:, :, 0] = Q[:, :, 1] = 1 / np.sqrt(2)

    pareto_archive = []
    history = {'best_ct': [], 'best_load_std': [], 'best_qloss': [], 'archive_size': []}

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

        if progress > 0.7:
            improved_archive = []
            for sol, obj in pareto_archive:
                improved_sol, improved_obj = local_qloss_improvement(sol, obj)
                improved_archive.append((improved_sol, improved_obj))
            pareto_archive = improved_archive

        if pareto_archive:
            archive_objs = np.array([obj for _, obj in pareto_archive])
            history['best_ct'].append(archive_objs[:, 0].min())
            history['best_load_std'].append(archive_objs[:, 1].min())
            history['best_qloss'].append(archive_objs[:, 2].min())
        else:
            history['best_ct'].append(np.inf)
            history['best_load_std'].append(np.inf)
            history['best_qloss'].append(np.inf)
        history['archive_size'].append(len(pareto_archive))

        guided_archive = [(sol, pen_obj) for sol, pen_obj in zip(solutions, penalized_objs)]
        Q = update_Q(Q, guided_archive, X_obs, t, max_iter, mutation_mode)

        if t % 10 == 0:
            print(f"{mutation_mode} | Iter {t:3d}/{max_iter:3d} | "
                  f"Archive: {len(pareto_archive):2d} | "
                  f"CT: {history['best_ct'][-1]:6.2f} | "
                  f"STD: {history['best_load_std'][-1]:6.2f} | "
                  f"QLoss: {history['best_qloss'][-1]:6.4f}")

    return pareto_archive, history


# ==== 8. 辅助工具函数 ====
def has_empty_station(solution) -> bool:
    """判断解是否存在空工位（所有工位必须有任务）"""
    _, station_sequences, _ = solution
    return any(len(seq) == 0 for seq in station_sequences)


def compute_station_times(solution):
    """计算每个工位的总负载（含工具切换成本）"""
    station_assignment, station_sequences, tool_assignment = solution
    station_times = np.zeros(NUM_STATIONS, dtype=float)
    for s in range(NUM_STATIONS):
        seq = station_sequences[s]
        if not seq:
            continue
        station_times[s] += TASK_TIMES[seq[0]]
        prev_tool = tool_assignment[seq[0]]
        for t in seq[1:]:
            curr_tool = tool_assignment[t]
            station_times[s] += TASK_TIMES[t]
            if curr_tool != prev_tool:
                station_times[s] += TOOL_SWITCH_COST_MATRIX[prev_tool, curr_tool]
            prev_tool = curr_tool
    return station_times


# ==== 9. 性能对比可视化模块 ====
def plot_comparison_results(results):
    """绘制四种变异模式的性能对比图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建4x2的子图布局
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    # fig.suptitle('四种变异模式性能对比', fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for idx, (mode_name, mode_results) in enumerate(results.items()):
        color = colors[idx]
        marker = markers[idx]

        # 提取5次运行的历史数据
        all_histories = mode_results['histories']

        # 计算平均历史数据
        avg_best_ct = np.mean([h['best_ct'] for h in all_histories], axis=0)
        avg_best_std = np.mean([h['best_load_std'] for h in all_histories], axis=0)
        avg_best_qloss = np.mean([h['best_qloss'] for h in all_histories], axis=0)
        avg_archive_size = np.mean([h['archive_size'] for h in all_histories], axis=0)

        iterations = range(len(avg_best_ct))

        # 子图1：CT收敛曲线
        axes[0, 0].plot(iterations, avg_best_ct, color=color, marker=marker,
                        markersize=3, linewidth=2, label=mode_name, alpha=0.8)
        axes[0, 0].set_title('CT convergence curve')
        axes[0, 0].set_ylabel('CT (s)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # 子图2：LoadSTD收敛曲线
        axes[0, 1].plot(iterations, avg_best_std, color=color, marker=marker,
                        markersize=3, linewidth=2, label=mode_name, alpha=0.8)
        axes[0, 1].set_title('LoadSTD convergence curve')
        axes[0, 1].set_ylabel('LoadSTD (s)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # 子图3：QLoss收敛曲线
        axes[1, 0].plot(iterations, avg_best_qloss, color=color, marker=marker,
                        markersize=3, linewidth=2, label=mode_name, alpha=0.8)
        axes[1, 0].set_title('QLoss convergence curve')
        axes[1, 0].set_ylabel('QLoss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # 子图4：档案大小变化
        axes[1, 1].plot(iterations, avg_archive_size, color=color, marker=marker,
                        markersize=3, linewidth=2, label=mode_name, alpha=0.8)
        axes[1, 1].set_title('帕累托档案大小变化')
        axes[1, 1].set_ylabel('解数量')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        # 子图5：最终解的质量分布（箱线图数据准备）
        final_ct_values = []
        final_std_values = []
        final_qloss_values = []

        for run_idx in range(NUM_RUNS):
            pareto_archive = mode_results['pareto_archives'][run_idx]
            if pareto_archive:
                ct_values = [obj[0] for _, obj in pareto_archive]
                std_values = [obj[1] for _, obj in pareto_archive]
                qloss_values = [obj[2] for _, obj in pareto_archive]

                # 取每个目标的前5个最优解
                final_ct_values.extend(sorted(ct_values)[:5])
                final_std_values.extend(sorted(std_values)[:5])
                final_qloss_values.extend(sorted(qloss_values)[:5])

        # 箱线图位置计算
        pos = idx + 1

        # 子图5：CT最终解分布
        if final_ct_values:
            axes[2, 0].boxplot([final_ct_values], positions=[pos], widths=0.6,
                               patch_artist=True,
                               boxprops=dict(facecolor=color, alpha=0.7),
                               medianprops=dict(color='black', linewidth=2))

        # 子图6：LoadSTD最终解分布
        if final_std_values:
            axes[2, 1].boxplot([final_std_values], positions=[pos], widths=0.6,
                               patch_artist=True,
                               boxprops=dict(facecolor=color, alpha=0.7),
                               medianprops=dict(color='black', linewidth=2))

        # 子图7：QLoss最终解分布
        if final_qloss_values:
            axes[3, 0].boxplot([final_qloss_values], positions=[pos], widths=0.6,
                               patch_artist=True,
                               boxprops=dict(facecolor=color, alpha=0.7),
                               medianprops=dict(color='black', linewidth=2))

    # 设置箱线图坐标轴
    axes[2, 0].set_title('Final CT Distribution')
    axes[2, 0].set_ylabel('CT (s)')
    axes[2, 0].set_xticks(range(1, len(MUTATION_MODES) + 1))
    axes[2, 0].set_xticklabels([name.split(' ')[0] for name in MUTATION_MODES.keys()])
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].set_title('Final LoadSTD Distribution')
    axes[2, 1].set_ylabel('LoadSTD (s)')
    axes[2, 1].set_xticks(range(1, len(MUTATION_MODES) + 1))
    axes[2, 1].set_xticklabels([name.split(' ')[0] for name in MUTATION_MODES.keys()])
    axes[2, 1].grid(True, alpha=0.3)

    axes[3, 0].set_title('Final QLoss Distribution')
    axes[3, 0].set_ylabel('QLoss')
    axes[3, 0].set_xticks(range(1, len(MUTATION_MODES) + 1))
    axes[3, 0].set_xticklabels([name.split(' ')[0] for name in MUTATION_MODES.keys()])
    axes[3, 0].grid(True, alpha=0.3)

    # 隐藏最后一个子图
    axes[3, 1].axis('off')

    plt.tight_layout()
    plt.show()


def plot_statistical_comparison(results):
    """绘制统计性能对比图（柱状图+误差线）"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

    # 计算各模式的统计指标
    modes = list(MUTATION_MODES.keys())
    avg_final_ct = []
    std_final_ct = []
    avg_final_std = []
    std_final_std = []
    avg_final_qloss = []
    std_final_qloss = []
    avg_archive_size = []

    for mode_name in modes:
        mode_results = results[mode_name]

        # 计算最终代的CT、STD、QLoss平均值和标准差
        final_ct_values = []
        final_std_values = []
        final_qloss_values = []
        archive_sizes = []

        for run_idx in range(NUM_RUNS):
            history = mode_results['histories'][run_idx]
            # 取最后10代的平均值作为最终性能
            if len(history['best_ct']) >= 10:
                final_ct_values.append(np.mean(history['best_ct'][-10:]))
                final_std_values.append(np.mean(history['best_load_std'][-10:]))
                final_qloss_values.append(np.mean(history['best_qloss'][-10:]))
                archive_sizes.append(np.mean(history['archive_size'][-10:]))

        avg_final_ct.append(np.mean(final_ct_values))
        std_final_ct.append(np.std(final_ct_values))
        avg_final_std.append(np.mean(final_std_values))
        std_final_std.append(np.std(final_std_values))
        avg_final_qloss.append(np.mean(final_qloss_values))
        std_final_qloss.append(np.std(final_qloss_values))
        avg_archive_size.append(np.mean(archive_sizes))

    # 创建柱状图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # fig.suptitle('四种变异模式统计性能对比', fontsize=16, fontweight='bold')

    x_pos = np.arange(len(modes))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # 子图1：最终CT对比
    bars1 = axes[0, 0].bar(x_pos, avg_final_ct, yerr=std_final_ct,
                           capsize=5, alpha=0.7, color=colors)
    axes[0, 0].set_title('Average CT')
    axes[0, 0].set_ylabel('CT (s)')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([name.split(' ')[0] for name in modes])

    # 在柱子上添加数值标签
    for bar, value in zip(bars1, avg_final_ct):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

    # 子图2：最终LoadSTD对比
    bars2 = axes[0, 1].bar(x_pos, avg_final_std, yerr=std_final_std,
                           capsize=5, alpha=0.7, color=colors)
    axes[0, 1].set_title('Average LoadSTD')
    axes[0, 1].set_ylabel('LoadSTD (s)')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([name.split(' ')[0] for name in modes])

    for bar, value in zip(bars2, avg_final_std):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    # 子图3：最终QLoss对比
    bars3 = axes[1, 0].bar(x_pos, avg_final_qloss, yerr=std_final_qloss,
                           capsize=5, alpha=0.7, color=colors)
    axes[1, 0].set_title('Average QLoss')
    axes[1, 0].set_ylabel('QLoss')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([name.split(' ')[0] for name in modes])

    for bar, value in zip(bars3, avg_final_qloss):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')


    plt.tight_layout()
    plt.show()


# ==== 10. 主实验程序 ====
def run_comparison_experiment():
    """运行四种变异模式的对比实验"""
    print("=" * 70)
    print("焊装线三目标IQEA算法 - 变异模式对比实验")
    print(f"参数配置：种群规模={POP_SIZE} | 最大迭代次数={MAX_ITER}")
    print(f"运行设置：每种模式运行{NUM_RUNS}次")
    print("=" * 70)

    # 存储所有结果
    all_results = {}

    for mode_name in MUTATION_MODES.keys():
        print(f"\n>>> 开始测试变异模式: {mode_name}")
        print("-" * 50)

        mode_results = {
            'histories': [],
            'pareto_archives': []
        }

        for run in range(NUM_RUNS):
            print(f"第{run + 1}次运行...")
            pareto_archive, history = quantum_evolutionary_optimization(
                pop_size=POP_SIZE,
                max_iter=MAX_ITER,
                n_obs_base=N_OBS_BASE,
                mutation_mode=mode_name
            )

            mode_results['histories'].append(history)
            mode_results['pareto_archives'].append(pareto_archive)

        all_results[mode_name] = mode_results
        print(f"✓ {mode_name} 模式完成 {NUM_RUNS} 次运行")

    # 绘制对比结果
    print("\n" + "=" * 70)
    print("开始生成性能对比图表...")
    print("=" * 70)

    # 绘制详细对比图
    plot_comparison_results(all_results)

    # 绘制统计对比图
    plot_statistical_comparison(all_results)

    # 打印总结统计
    print_summary_statistics(all_results)


def print_summary_statistics(results):
    """打印四种模式的总结统计信息"""
    print("\n" + "=" * 80)
    print("四种变异模式性能总结统计")
    print("=" * 80)

    for mode_name in MUTATION_MODES.keys():
        mode_results = results[mode_name]

        # 计算关键指标
        final_ct_values = []
        final_std_values = []
        final_qloss_values = []

        for run_idx in range(NUM_RUNS):
            history = mode_results['histories'][run_idx]
            if len(history['best_ct']) >= 10:
                final_ct_values.append(np.mean(history['best_ct'][-10:]))
                final_std_values.append(np.mean(history['best_load_std'][-10:]))
                final_qloss_values.append(np.mean(history['best_qloss'][-10:]))

        # 计算统计量
        avg_ct = np.mean(final_ct_values)
        std_ct = np.std(final_ct_values)
        avg_std = np.mean(final_std_values)
        std_std = np.std(final_std_values)
        avg_qloss = np.mean(final_qloss_values)
        std_qloss = np.std(final_qloss_values)

        print(f"\n{mode_name}:")
        print(f"  生产节拍CT: {avg_ct:.2f} ± {std_ct:.2f} 秒")
        print(f"  负载标准差: {avg_std:.2f} ± {std_std:.2f} 秒")
        print(f"  质量损失QLoss: {avg_qloss:.4f} ± {std_qloss:.4f}")

    print("\n" + "=" * 80)


# ==== 主程序入口 ====
if __name__ == "__main__":
    # 运行对比实验
    run_comparison_experiment()
