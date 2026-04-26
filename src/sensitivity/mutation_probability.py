import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""焊装线三目标IQEA算法（变异概率实验版）
目标：最小化生产节拍(CT)、最小化工位负载标准差(LoadSTD)、最小化质量损失(QLoss)
功能：测试不同变异概率上限对算法性能的影响，超体积计算采用min-max归一化优化
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
MAX_ARCHIVE_SIZE = 20  # 帕累托档案最大大小

# ==== 质量相关参数 ====
QUAL_ALPHA = 0.2  # 粗糙度权重系数
QUAL_BETA = 0.5  # 缺陷率权重系数
QUAL_GAMMA = 0.3  # 膨胀量权重系数
TARGET_EXPANSION = 0.025  # 目标膨胀量
POSITIVE_WEIGHT = 1.5  # 膨胀量正偏差权重
NEGATIVE_WEIGHT = 0.8  # 膨胀量负偏差权重

# ==== CT阈值参数 ====
MAX_CT_THRESHOLD = 600.0  # 生产节拍最大允许阈值

# ==== 超体积计算专用：固定参考点（确保不同运行间可比性）====
HV_FIXED_REF_POINT = np.array([600.0, 100.0, 1.0])  # [CT_max, LoadSTD_max, QLoss_max]


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
    result = QUAL_ALPHA * roughness_loss + QUAL_BETA * defect_loss + QUAL_GAMMA * expansion_loss
    print(f"总质量损失: {result:.6f}")
    return result


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
                penalty += 5.0  # 空工位施加惩罚

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

    # 限制档案大小（超出时保留分布更均匀的解）
    if len(new_archive) > MAX_ARCHIVE_SIZE:
        # 简单随机截断（实际应用中可使用更智能的归档策略）
        new_archive = random.sample(new_archive, MAX_ARCHIVE_SIZE)

    return new_archive


def select_representative_solutions(pareto_archive):
    """从帕累托档案中选择参考解（理想点、膝点）"""
    reps = []
    if not pareto_archive:
        return reps

    # 目标值归一化
    objs = np.array([obj for _, obj in pareto_archive], dtype=float)
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    ranges = np.maximum(maxs - mins, EPS)
    norm_objs = (objs - mins) / ranges

    # 选择理想点（到归一化理想点[0,0,0]距离最近）
    utopia = np.zeros(3)
    d2utopia = np.linalg.norm(norm_objs - utopia, axis=1)
    idx_ideal = int(np.argmin(d2utopia))
    reps.append(("Ideal-point", pareto_archive[idx_ideal]))

    # 选择膝点（到理想点→nadir点直线垂直距离最大）
    nadir = np.ones(3)
    line_dir = (nadir - utopia) / np.linalg.norm(nadir - utopia)
    projs = (norm_objs @ line_dir.reshape(-1, 1)) * line_dir.reshape(1, -1)
    perp_vecs = norm_objs - projs
    dist_line = np.linalg.norm(perp_vecs, axis=1)
    idx_knee = int(np.argmax(dist_line))
    reps.append(("Knee-point", pareto_archive[idx_knee]))
    return reps


# ==== 5. 量子进化更新模块 ====
def update_Q(Q, guided_archive, X_obs, t, max_iter, p_mut_max=0.06):
    """更新量子种群的量子态（仅基于有效解更新）"""
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
    p_mut_min = p_mut_max / 10.0
    size_factor = 1.0 / (1.0 + np.log1p(max(archive_size, 1)))
    p_mut = p_mut_min + (p_mut_max - p_mut_min) * (1 - progress) * size_factor
    phi_max = (0.08 * np.pi) * (1 - progress) + 0.02 * np.pi
    p_reinit, p_flip = 0.5 * p_mut, 0.5 * p_mut

    # 仅对有效解对应的量子个体更新（避免索引超出）
    update_indices = np.random.choice(pop_size, size=valid_size, replace=False)
    for idx in range(valid_size):
        j = update_indices[idx]  # 量子个体索引（在pop_size范围内）
        for i in range(num_qubits):
            alpha, beta = Q[j, i]
            xi = X_obs[idx, i]  # 有效解对应的观测比特（idx在valid_size范围内）
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
            boost = 1.0 + 1.5 * max(0.0, polarization - 0.35)
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

    return new_Q


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
def quantum_evolutionary_optimization(pop_size=30, max_iter=100, n_obs_base=3, p_mut_max=0.06):
    """三目标量子进化算法主循环，返回(帕累托档案, 优化历史记录)"""
    # 初始化量子种群（等概率状态 |0⟩+|1⟩/√2）
    num_qubits = NUM_TASKS * (NUM_STATIONS + NUM_TOOL_TYPES + 1)
    Q = np.zeros((pop_size, num_qubits, 2))
    Q[:, :, 0] = Q[:, :, 1] = 1 / np.sqrt(2)

    # 初始化档案与历史记录
    pareto_archive = []
    history = {'best_ct': [], 'best_load_std': [], 'best_qloss': [], 'archive_size': []}

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
        Q = update_Q(Q, guided_archive, X_obs, t, max_iter, p_mut_max)

        # 打印迭代进度
        if t % 10 == 0:
            print(f"IQEA Iter {t:3d}/{max_iter:3d} | "
                  f"Archive Size: {len(pareto_archive):2d} | "
                  f"Valid Solutions: {len(guided_archive):2d} | "
                  f"Best CT: {history['best_ct'][-1]:6.2f} | "
                  f"Best LoadSTD: {history['best_load_std'][-1]:6.2f} | "
                  f"Best QLoss: {history['best_qloss'][-1]:6.4f} | "
                  f"p_mut_max: {p_mut_max:.3f}")

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
def run_mutation_experiment(p_mut_max_values, num_runs=20, pop_size=30, max_iter=100):
    """运行变异概率实验，收集性能数据（包括迭代历史）"""
    results = []
    convergence_history = {}  # 存储收敛历史数据

    for p_mut_max in p_mut_max_values:
        print(f"\n=== 开始测试变异概率上限: {p_mut_max} ===")
        convergence_history[p_mut_max] = {
            'ct': np.zeros((num_runs, max_iter)),
            'std': np.zeros((num_runs, max_iter)),
            'qloss': np.zeros((num_runs, max_iter)),
            'hv': np.zeros((num_runs, max_iter))
        }

        for run in range(num_runs):
            print(f"运行 {run + 1}/{num_runs}...")

            # 运行算法
            np.random.seed(run)  # 设置随机种子以确保可重复性
            random.seed(run)
            pareto_archive, history = quantum_evolutionary_optimization(
                pop_size=pop_size,
                max_iter=max_iter,
                p_mut_max=p_mut_max
            )

            # 记录收敛历史
            convergence_history[p_mut_max]['ct'][run] = history['best_ct']
            convergence_history[p_mut_max]['std'][run] = history['best_load_std']
            convergence_history[p_mut_max]['qloss'][run] = history['best_qloss']
            convergence_history[p_mut_max]['hv'][run] = history.get('hypervolume', [0] * max_iter)

            # 收集最终性能指标
            if pareto_archive:
                archive_objs = np.array([obj for _, obj in pareto_archive])
                best_ct = archive_objs[:, 0].min()
                best_std = archive_objs[:, 1].min()
                best_qloss = archive_objs[:, 2].min()
                hypervolume = calculate_hypervolume(archive_objs)
            else:
                best_ct, best_std, best_qloss, hypervolume = np.inf, np.inf, np.inf, 0

            # 存储结果
            results.append({
                'p_mut_max': p_mut_max,
                'run': run,
                'best_ct': best_ct,
                'best_std': best_std,
                'best_qloss': best_qloss,
                'hypervolume': hypervolume
            })

    return pd.DataFrame(results), convergence_history


def calculate_hypervolume(objectives):
    """计算超体积指标（使用min-max归一化优化）"""
    if len(objectives) == 0:
        return 0.0

    # 转换为numpy数组
    objs = np.array(objectives, dtype=np.float64)

    # 计算当前帕累托前沿的最小值（用于min-max归一化）
    objs_min = objs.min(axis=0)

    # 检查参考点是否有效（必须所有目标都大于等于前沿最大值）
    for i in range(3):
        if HV_FIXED_REF_POINT[i] < objs[:, i].max():
            # 若参考点不够差，动态扩展（避免归一化后出现负值）
            HV_FIXED_REF_POINT[i] = objs[:, i].max() * 1.1

    # min-max归一化：(x - x_min) / (ref_x - x_min)，将目标值映射到[0,1]
    # 分母添加EPS避免除零
    norm_objs = (objs - objs_min) / (HV_FIXED_REF_POINT - objs_min + EPS)

    # 确保归一化后的值在[0,1]范围内（处理数值误差）
    norm_objs = np.clip(norm_objs, 0.0, 1.0)

    # 3D超体积计算：按CT排序后累加体积
    # 1. 按第一个目标（CT）升序排序
    sorted_indices = np.argsort(norm_objs[:, 0])
    sorted_objs = norm_objs[sorted_indices]

    # 2. 初始化超体积
    hypervolume = 0.0
    num_points = len(sorted_objs)

    # 3. 累加每个点贡献的体积
    for i in range(num_points):
        # 当前点的坐标
        x, y, z = sorted_objs[i]

        # 计算当前点在y-z平面的"有效面积"（参考点为(1,1,1)）
        area = (1.0 - y) * (1.0 - z)

        # 计算x方向的长度（到下一个点的距离，最后一个点到参考点）
        if i < num_points - 1:
            dx = sorted_objs[i + 1, 0] - x
        else:
            dx = 1.0 - x  # 最后一个点到参考点的距离

        # 累加体积（面积 × 长度）
        hypervolume += area * dx

    return hypervolume


def plot_mutation_performance_comparison(results_df, convergence_history):
    """绘制超体积和收敛速度性能比较图"""
    # 使用seaborn设置样式和调色板
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 超体积比较
    sns.boxplot(x='p_mut_max', y='hypervolume', data=results_df, ax=axes[0])
    axes[0].set_title('超体积(Hypervolume)比较', fontsize=12)
    axes[0].set_ylabel('Hypervolume', fontsize=10)
    axes[0].set_xlabel('变异概率上限', fontsize=10)
    axes[0].grid(alpha=0.3)

    # 收敛速度比较（CT随迭代次数的变化）
    max_iter = len(next(iter(convergence_history.values()))['ct'][0])
    iterations = range(1, max_iter + 1)

    for p_mut_max in convergence_history:
        # 计算每个迭代步的平均CT
        avg_ct = np.mean(convergence_history[p_mut_max]['ct'], axis=0)
        axes[1].plot(iterations, avg_ct, label=f'p_mut_max={p_mut_max}', linewidth=2)

    axes[1].set_title('收敛速度比较 (生产节拍CT)', fontsize=12)
    axes[1].set_ylabel('平均CT (秒)', fontsize=10)
    axes[1].set_xlabel('迭代次数', fontsize=10)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_mutation_trend_analysis(results_df, convergence_history):
    """绘制变异概率趋势分析图（超体积和收敛速度）"""
    # 使用seaborn设置样式和调色板
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 计算每个变异概率的平均性能
    summary = results_df.groupby('p_mut_max').agg({
        'best_ct': 'mean',
        'best_std': 'mean',
        'best_qloss': 'mean',
        'hypervolume': 'mean'
    }).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 收敛速度趋势（CT, LoadSTD, QLoss）
    max_iter = len(next(iter(convergence_history.values()))['ct'][0])
    iterations = range(1, max_iter + 1)

    # 计算每个变异概率的平均收敛曲线
    for p_mut_max in convergence_history:
        # 计算每个迭代步的平均值
        avg_ct = np.mean(convergence_history[p_mut_max]['ct'], axis=0)
        avg_std = np.mean(convergence_history[p_mut_max]['std'], axis=0)
        avg_qloss = np.mean(convergence_history[p_mut_max]['qloss'], axis=0)

        # 绘制CT收敛曲线
        axes[0].plot(iterations, avg_ct, label=f'CT (p_mut_max={p_mut_max})', linewidth=2)
        # 绘制LoadSTD收敛曲线（使用不同线型）
        axes[0].plot(iterations, avg_std, '--', label=f'LoadSTD (p_mut_max={p_mut_max})', linewidth=2)
        # 绘制QLoss收敛曲线（使用不同线型）
        axes[0].plot(iterations, avg_qloss, ':', label=f'QLoss (p_mut_max={p_mut_max})', linewidth=2)

    axes[0].set_title('收敛速度随变异概率变化趋势', fontsize=12)
    axes[0].set_ylabel('目标值', fontsize=10)
    axes[0].set_xlabel('迭代次数', fontsize=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 超体积趋势
    axes[1].plot(summary['p_mut_max'], summary['hypervolume'], 'o-', linewidth=2, markersize=8)
    axes[1].set_title('超体积(Hypervolume)随变异概率变化趋势', fontsize=12)
    axes[1].set_ylabel('平均Hypervolume', fontsize=10)
    axes[1].set_xlabel('变异概率上限', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==== 10. 主程序（入口） ====
if __name__ == "__main__":
    # 算法参数
    POP_SIZE = 30
    MAX_ITER = 100
    N_OBS_BASE = 3

    # 变异概率实验参数（可根据需要调整）
    P_MUT_MAX_VALUES = [0.04, 0.06, 0.08, 0.10]
    NUM_RUNS = 5  # 每个变异概率运行次数（调试时可设小，正式实验设20+）

    print("=" * 80)
    print("焊装线三目标IQEA算法 - 变异概率上限实验")
    print(f"测试变异概率上限: {P_MUT_MAX_VALUES}")
    print(f"每个概率运行次数: {NUM_RUNS}")
    print("=" * 80)

    # 运行变异概率实验
    results_df, convergence_history = run_mutation_experiment(
        p_mut_max_values=P_MUT_MAX_VALUES,
        num_runs=NUM_RUNS,
        pop_size=POP_SIZE,
        max_iter=MAX_ITER
    )

    # 保存实验结果
    results_df.to_csv('mutation_probability_experiment_results.csv', index=False)
    print("实验结果已保存到 mutation_probability_experiment_results.csv")

    # 绘制性能比较图
    print("\n绘制性能比较图...")
    plot_mutation_performance_comparison(results_df, convergence_history)

    # 绘制趋势分析图
    print("绘制趋势分析图...")
    plot_mutation_trend_analysis(results_df, convergence_history)

    print("\n实验完成！")
