import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

"""焊装线三目标IQEA算法（极化阈值实验版）
目标：最小化生产节拍(CT)、最小化工位负载标准差(LoadSTD)、最小化质量损失(QLoss)
新增功能：测试不同极化阈值对算法性能的影响
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
# QLoss不设ε容差，需要严格更小

# ==== 档案大小限制 ====
MAX_ARCHIVE_SIZE = 20  

# ==== 质量相关参数 ====
QUAL_ALPHA = 0.2  # 粗糙度权重系数
QUAL_BETA = 0.5  # 缺陷率权重系数
QUAL_GAMMA = 0.3  # 膨胀量权重系数
TARGET_EXPANSION = 0.025  # 目标膨胀量
POSITIVE_WEIGHT = 1.5  # 膨胀量正偏差权重
NEGATIVE_WEIGHT = 0.8  # 膨胀量负偏差权重

# ==== CT阈值参数 ====
MAX_CT_THRESHOLD = 600.0  # 生产节拍最大允许阈值


# ==== 1. 质量损失计算模块 ====
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
                penalty += 0.0  # 空工位施加惩罚5分

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
    # CT和STD使用ε容差比较，QLoss必须严格更小
    ct_dominated = a[0] <= b[0] + EPS_CT
    std_dominated = a[1] <= b[1] + EPS_STD
    qloss_dominated = a[2] < b[2]  # QLoss必须严格更小

    # 检查是否所有目标都ε-支配
    all_dominated = ct_dominated and std_dominated and qloss_dominated

    # 检查是否至少有一个目标严格ε-支配
    ct_strict = a[0] < b[0] - EPS_CT
    std_strict = a[1] < b[1] - EPS_STD
    qloss_strict = a[2] < b[2]  # QLoss已经要求严格更小

    any_strict = ct_strict or std_strict or qloss_strict

    return all_dominated and any_strict


def update_archive(archive, cand_solution, cand_obj):
    """更新帕累托档案，使用ε-支配"""
    # 如果候选解CT超过阈值，直接返回原档案
    if cand_obj[0] > MAX_CT_THRESHOLD:
        return archive

    # 候选解被档案中任意解ε-支配则丢弃
    for _, obj in archive:
        if dominates(obj, cand_obj):
            return archive

    # 移除档案中被候选解ε-支配的解
    new_archive = [(sol, obj) for sol, obj in archive if not dominates(cand_obj, obj)]

    # 加入候选解
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


# ==== 5. 量子进化更新模块 ====
def update_Q(Q, guided_archive, X_obs, t, max_iter, polarization_threshold=0.35):
    pop_size, num_qubits, _ = Q.shape
    new_Q = np.copy(Q)
    valid_size = len(guided_archive)
    if valid_size == 0:
        return new_Q  # 无有效解时不更新
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
        # 默认参考解（任务均匀分配）
        ref_station = np.array([i % NUM_STATIONS for i in range(NUM_TASKS)])
        ref_tool = np.array([min(ALLOWED_TOOLS[i]) for i in range(NUM_TASKS)])
        ref_seq = [[] for _ in range(NUM_STATIONS)]
        for i, s in enumerate(ref_station):
            ref_seq[s].append(i)
        ref_solution = (ref_station, ref_seq, ref_tool)

    # 编码参考解为比特串
    ref_bits = []
    station_assignment, station_sequences, tool_assignment = ref_solution
    # 编码任务-工位分配
    for task in range(NUM_TASKS):
        cs = station_assignment[task]
        for s in range(NUM_STATIONS):
            ref_bits.append(1 if s == cs else 0)
    # 编码任务-工具分配
    for task in range(NUM_TASKS):
        ct = tool_assignment[task]
        for tt in range(NUM_TOOL_TYPES):
            ref_bits.append(1 if tt == ct else 0)
    # 编码工位内排序
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
    phi_max = (0.08 * np.pi) * (1 - progress) + 0.02 * np.pi
    p_reinit, p_flip = 0.5 * p_mut, 0.5 * p_mut

    # 记录极化度统计信息
    polarization_stats = {
        'total_bits': 0,
        'above_threshold': 0,
        'polarization_values': []
    }

    update_indices = np.random.choice(pop_size, size=valid_size, replace=False)
    for idx in range(valid_size):
        j = update_indices[idx]  # 量子个体索引
        for i in range(num_qubits):
            alpha, beta = Q[j, i]
            xi = X_obs[idx, i]  # 有效解对应的观测比特
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
            # 归一化
            norm = np.hypot(new_alpha, new_beta)
            if norm < EPS:
                new_alpha, new_beta = alpha, beta
                norm = np.hypot(new_alpha, new_beta)
            new_alpha /= norm
            new_beta /= norm

            # 自适应量子变异
            p1 = float(abs(new_beta) ** 2)
            polarization = abs(p1 - 0.5)

            # 记录极化度统计信息
            polarization_stats['total_bits'] += 1
            polarization_stats['polarization_values'].append(polarization)
            if polarization > polarization_threshold:
                polarization_stats['above_threshold'] += 1

            boost = 1.0 + 1.5 * max(0.0, polarization - polarization_threshold)
            if np.random.rand() < p_mut * boost:
                r = np.random.rand()
                if r < p_reinit:
                    # 量子态重置
                    new_alpha = new_beta = 1 / np.sqrt(2)
                elif r < p_reinit + p_flip:
                    # 量子态翻转+微小扰动
                    new_alpha, new_beta = new_beta, new_alpha
                    jitter = np.random.uniform(-0.01 * np.pi, 0.01 * np.pi)
                    c2, s2 = np.cos(jitter), np.sin(jitter)
                    new_alpha, new_beta = c2 * new_alpha - s2 * new_beta, s2 * new_alpha + c2 * new_beta
                else:
                    # 量子态小扰动
                    delta = np.random.uniform(-phi_max, phi_max)
                    c2, s2 = np.cos(delta), np.sin(delta)
                    new_alpha, new_beta = c2 * new_alpha - s2 * new_beta, s2 * new_alpha + c2 * new_beta
                # 再次归一化
                norm = np.hypot(new_alpha, new_beta)
                if norm < EPS:
                    new_alpha, new_beta = 1 / np.sqrt(2), 1 / np.sqrt(2)
                else:
                    new_alpha /= norm
                    new_beta /= norm

            new_Q[j, i, 0] = new_alpha
            new_Q[j, i, 1] = new_beta

    return new_Q, polarization_stats


# ==== 6. QLoss局部改进模块 ====
def local_qloss_improvement(solution, obj_values):
    """在不恶化CT和LoadSTD的前提下，通过微调工具分配来改进QLoss"""
    station_assignment, station_sequences, tool_assignment = solution
    ct, load_std, qloss = obj_values

    improved = False
    new_tool_assignment = tool_assignment.copy()

    # 尝试对每个任务微调工具
    for task in range(NUM_TASKS):
        original_tool = tool_assignment[task]
        allowed_tools = list(ALLOWED_TOOLS[task])

        # 如果只有一个可选工具，跳过
        if len(allowed_tools) <= 1:
            continue

        # 尝试所有可能的工具更换
        for new_tool in allowed_tools:
            if new_tool == original_tool:
                continue

            # 创建新的工具分配
            test_tool_assignment = new_tool_assignment.copy()
            test_tool_assignment[task] = new_tool

            # 创建新解
            test_solution = (station_assignment, station_sequences, test_tool_assignment)

            # 评估新解
            test_obj, _, _ = evaluate_welding_objectives_with_penalty(test_solution)
            test_ct, test_std, test_qloss = test_obj

            # 检查是否满足改进条件：CT和LoadSTD不恶化，QLoss降低
            # 使用更严格的条件：CT和LoadSTD不能增加，QLoss必须降低
            ct_ok = test_ct <= ct  # 不允许CT增加
            std_ok = test_std <= load_std  # 不允许LoadSTD增加
            qloss_improved = test_qloss < qloss - EPS  # QLoss需要严格降低

            if ct_ok and std_ok and qloss_improved:
                # 接受改进
                new_tool_assignment = test_tool_assignment
                ct, load_std, qloss = test_ct, test_std, test_qloss
                improved = True
                break  # 找到一个改进就跳出内层循环

    # 如果有改进，返回新解和新目标值
    if improved:
        improved_solution = (station_assignment, station_sequences, new_tool_assignment)
        improved_obj = (ct, load_std, qloss)
        return improved_solution, improved_obj

    # 没有改进，返回原解
    return solution, obj_values


# ==== 7. IQEA主循环 ====
def quantum_evolutionary_optimization(pop_size=30, max_iter=100, n_obs_base=3, polarization_threshold=0.35):
    """三目标量子进化算法主循环，返回(帕累托档案, 优化历史记录, 极化度统计)"""
    # 初始化量子种群（等概率状态 |0⟩+|1⟩/√2）
    num_qubits = NUM_TASKS * (NUM_STATIONS + NUM_TOOL_TYPES + 1)
    Q = np.zeros((pop_size, num_qubits, 2))
    Q[:, :, 0] = Q[:, :, 1] = 1 / np.sqrt(2)

    # 初始化档案与历史记录
    pareto_archive = []
    history = {'best_ct': [], 'best_load_std': [], 'best_qloss': [], 'archive_size': []}
    polarization_history = []  # 记录每代的极化度统计

    # 迭代优化
    for t in range(max_iter):
        solutions = []
        original_objs = []
        penalized_objs = []
        X_obs = []
        progress = t / max_iter
        n_obs = n_obs_base + int(2 * progress) if progress < 1.0 else n_obs_base + 2

        # 观测每个量子个体
        for j in range(pop_size):
            sol, obs_bits, orig_obj = observe_individual(Q[j], n_obs)
            _, penalized_obj, _ = evaluate_welding_objectives_with_penalty(sol)
            if not has_empty_station(sol):  # 过滤空站解
                solutions.append(sol)
                original_objs.append(orig_obj)
                penalized_objs.append(penalized_obj)
                X_obs.append(obs_bits)

        X_obs = np.array(X_obs) if X_obs else np.empty((0, num_qubits))

        # 更新帕累托档案
        for sol, orig_obj in zip(solutions, original_objs):
            if not np.any(np.isinf(orig_obj)):
                pareto_archive = update_archive(pareto_archive, sol, orig_obj)

        # 在后期迭代中应用QLoss局部改进
        if progress > 0.7:  # 最后30%的迭代
            improved_archive = []
            for sol, obj in pareto_archive:
                improved_sol, improved_obj = local_qloss_improvement(sol, obj)
                improved_archive.append((improved_sol, improved_obj))
            pareto_archive = improved_archive

        # 记录历史数据
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

        # 更新量子种群
        guided_archive = [(sol, pen_obj) for sol, pen_obj in zip(solutions, penalized_objs)]
        Q, polarization_stats = update_Q(Q, guided_archive, X_obs, t, max_iter, polarization_threshold)
        polarization_history.append(polarization_stats)

        # 打印迭代进度
        if t % 10 == 0:
            print(f"IQEA Iter {t:3d}/{max_iter:3d} | "
                  f"Archive Size: {len(pareto_archive):2d} | "
                  f"Valid Solutions: {len(guided_archive):2d} | "
                  f"Best CT: {history['best_ct'][-1]:6.2f} | "
                  f"Best LoadSTD: {history['best_load_std'][-1]:6.2f} | "
                  f"Best QLoss: {history['best_qloss'][-1]:6.4f} | "
                  f"Polarization > {polarization_threshold}: {polarization_stats['above_threshold'] / max(polarization_stats['total_bits'], 1) * 100:.1f}%")

    return pareto_archive, history, polarization_history


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


def print_solution_details(solution, obj_values, title=""):
    """打印解的详细信息（工位分配、任务序列、目标值）"""
    station_assignment, station_sequences, tool_assignment = solution
    ct, load_std, qloss = obj_values
    station_times = compute_station_times(solution)

    if title:
        print("\n" + "=" * 50)
        print(f"【{title}】")
        print("=" * 50)
    print(f"三目标值：生产节拍CT={ct:.2f}秒 | 负载标准差LoadSTD={load_std:.2f}秒 | 质量损失QLoss={qloss:.4f}")
    print("工位任务分配详情：")
    for s in range(NUM_STATIONS):
        seq = station_sequences[s]
        if not seq:
            print(f"  工位{s}: 无任务 | 总负载: {station_times[s]:.1f}秒")
            continue
        task_str = " → ".join([f"任务{task + 1}(工具{tool_assignment[task]})" for task in seq])
        print(f"  工位{s}: {task_str} | 总负载: {station_times[s]:.1f}秒")
    print(f"最大工位负载（即生产节拍）: {np.max(station_times):.2f}秒")


# ==== 9. 实验分析与可视化模块 ====
def run_polarization_experiment(thresholds, num_runs=30, pop_size=30, max_iter=100):
    """运行极化阈值实验，收集性能数据"""
    results = []

    for threshold in thresholds:
        print(f"\n=== 开始测试极化阈值: {threshold} ===")

        for run in range(num_runs):
            print(f"运行 {run + 1}/{num_runs}...")

            # 运行算法
            np.random.seed(run)
            random.seed(run)
            pareto_archive, history, polarization_history = quantum_evolutionary_optimization(
                pop_size=pop_size,
                max_iter=max_iter,
                polarization_threshold=threshold
            )

            if pareto_archive:
                archive_objs = np.array([obj for _, obj in pareto_archive])
                best_ct = archive_objs[:, 0].min()
                best_std = archive_objs[:, 1].min()
                best_qloss = archive_objs[:, 2].min()
                hypervolume = calculate_hypervolume(archive_objs)
            else:
                best_ct, best_std, best_qloss, hypervolume = np.inf, np.inf, np.inf, 0

            total_polarization_above = 0
            total_polarization_bits = 0
            for stats in polarization_history:
                total_polarization_above += stats['above_threshold']
                total_polarization_bits += stats['total_bits']

            polarization_ratio = total_polarization_above / total_polarization_bits if total_polarization_bits > 0 else 0

            # 存储结果
            results.append({
                'threshold': threshold,
                'run': run,
                'best_ct': best_ct,
                'best_std': best_std,
                'best_qloss': best_qloss,
                'hypervolume': hypervolume,
                'polarization_ratio': polarization_ratio
            })

    return pd.DataFrame(results)


def calculate_hypervolume(objectives):
    if len(objectives) == 0:
        return 0

    # 设置参考点
    ref_point = np.max(objectives, axis=0) * 1.1

    normalized_objs = objectives / ref_point
    return np.prod(1 - normalized_objs, axis=1).sum()


def plot_performance_comparison(results_df):
    """绘制性能比较图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # CT比较
    sns.boxplot(x='threshold', y='best_ct', data=results_df, ax=axes[0, 0])
    axes[0, 0].set_title('生产节拍(CT)比较')
    axes[0, 0].set_ylabel('CT (秒)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # LoadSTD比较
    sns.boxplot(x='threshold', y='best_std', data=results_df, ax=axes[0, 1])
    axes[0, 1].set_title('负载标准差(LoadSTD)比较')
    axes[0, 1].set_ylabel('LoadSTD (秒)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # QLoss比较
    sns.boxplot(x='threshold', y='best_qloss', data=results_df, ax=axes[1, 0])
    axes[1, 0].set_title('质量损失(QLoss)比较')
    axes[1, 0].set_ylabel('QLoss')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 超体积比较
    sns.boxplot(x='threshold', y='hypervolume', data=results_df, ax=axes[1, 1])
    axes[1, 1].set_title('超体积(Hypervolume)比较')
    axes[1, 1].set_ylabel('Hypervolume')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_polarization_analysis(results_df):
    """绘制极化度分析图"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 极化度比例与性能关系
    for threshold in results_df['threshold'].unique():
        subset = results_df[results_df['threshold'] == threshold]
        axes[0].scatter(
            subset['polarization_ratio'],
            subset['hypervolume'],
            alpha=0.7,
            label=f'阈值={threshold}',
            s=50
        )

    axes[0].set_xlabel('极化度超过阈值的比例')
    axes[0].set_ylabel('超体积(Hypervolume)')
    axes[0].set_title('极化度与算法性能关系')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 不同阈值的极化度比例分布
    sns.boxplot(x='threshold', y='polarization_ratio', data=results_df, ax=axes[1])
    axes[1].set_title('不同阈值的极化度比例分布')
    axes[1].set_ylabel('极化度超过阈值的比例')
    axes[1].set_xlabel('极化阈值')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


# ==== 10. 主程序（入口） ====
if __name__ == "__main__":
    # 算法参数
    POP_SIZE = 30
    MAX_ITER = 100
    N_OBS_BASE = 3

    # 极化阈值实验参数
    POLARIZATION_THRESHOLDS = [0.25, 0.35, 0.45, 0.55]
    NUM_RUNS = 30  # 每个阈值运行30次

    print("=" * 80)
    print("焊装线三目标IQEA算法 - 极化阈值实验")
    print(f"测试阈值: {POLARIZATION_THRESHOLDS}")
    print(f"每个阈值运行次数: {NUM_RUNS}")
    print("=" * 80)

    # 运行极化阈值实验
    results_df = run_polarization_experiment(
        thresholds=POLARIZATION_THRESHOLDS,
        num_runs=NUM_RUNS,
        pop_size=POP_SIZE,
        max_iter=MAX_ITER
    )

    # 保存实验结果
    results_df.to_csv('polarization_threshold_experiment_results.csv', index=False)
    print("实验结果已保存到 polarization_threshold_experiment_results.csv")

    # 绘制性能比较图
    print("\n绘制性能比较图...")
    plot_performance_comparison(results_df)

    # 绘制极化度分析图
    print("绘制极化度分析图...")
    plot_polarization_analysis(results_df)

    print("\n实验完成！")
