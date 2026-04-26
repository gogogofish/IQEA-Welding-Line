import argparse
from pathlib import Path
from typing import Dict, Union

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import csv
import time
import platform
import sys
import json
import itertools
from scipy.stats import spearmanr
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

"""
焊装线三目标IQEA算法
目标：
    1) 最小化生产节拍 CT
    2) 最小化工位负载标准差 LoadSTD
    3) 最小化质量损失 QLoss
"""

# 基础随机种子
GLOBAL_RANDOM_SEED = 42
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)

# 基础参数
NUM_TASKS = 24
NUM_STATIONS = 6

TASK_TIMES = [
    55, 65, 55, 45, 45, 35, 55, 160, 35, 70,
    100, 80, 35, 60, 60, 160, 35, 70, 450, 40,
    35, 420, 40, 35
]

# 各任务允许使用的工具集合（0/1/2代表三种工具）
ALLOWED_TOOLS = []
for task_idx in range(NUM_TASKS):
    task_id = task_idx + 1
    if task_id in {1, 4, 5, 8, 9, 13, 17, 24}:
        ALLOWED_TOOLS.append({0})
    elif task_id in {2, 3, 6, 10}:
        ALLOWED_TOOLS.append({0, 1})
    else:
        ALLOWED_TOOLS.append({2})

NUM_TOOL_TYPES = 3

TOOL_SWITCH_COST_MATRIX = np.array([
    [0.0, 1.5, 1.8],
    [1.5, 0.0, 1.2],
    [1.8, 1.2, 0.0],
], dtype=float)

PRECEDENCE_CONSTRAINTS = [
    (0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (9, 11), (10, 11), (11, 12), (13, 15), (14, 15), (15, 16),
    (17, 18), (18, 19), (8, 20), (12, 20), (16, 20), (19, 20),
    (20, 21), (21, 22), (22, 23)
]

EPS = 1e-12

# ε-支配容差参数
GLOBAL_EPS_CT = 0.5
GLOBAL_EPS_STD = 0.25

LOCAL_EPS_CT = 5.0
LOCAL_EPS_STD = 3.0

# 档案大小限制
MAX_ARCHIVE_SIZE = 20

# 精英解参数
ELITE_TOP_K = 10
IDEAL_POINT_WEIGHTS = (0.4, 0.2, 0.4)

# 质量相关参数（归一化版）
QUAL_ALPHA = 0.2
QUAL_BETA = 0.5
QUAL_GAMMA = 0.3
BASE_WEIGHTS = np.array([QUAL_ALPHA, QUAL_BETA, QUAL_GAMMA], dtype=float)

TARGET_EXPANSION = 0.025
POSITIVE_WEIGHT = 1.5
NEGATIVE_WEIGHT = 0.8

# 归一化参数
R_MIN = 0.5
R_MAX = 2.0
D_MAX = 0.10
E_TOL = 0.15

BASE_QUALITY_PARAMS = {
    "R_min": R_MIN,
    "R_max": R_MAX,
    "D_max": D_MAX,
    "E_tol": E_TOL,
    "E_target": TARGET_EXPANSION,
    "w_plus": POSITIVE_WEIGHT,
    "w_minus": NEGATIVE_WEIGHT,
}

# 安全预计算缓存
TASK_TIMES_ARRAY = np.asarray(TASK_TIMES, dtype=float)
MAX_TASK_TIME = float(np.max(TASK_TIMES_ARRAY))
TASK_SCALE_CACHE = 1.0 + (TASK_TIMES_ARRAY / (MAX_TASK_TIME + EPS)) * 0.2
STATION_ADJUST_CACHE = np.asarray([1.0 + ((s % 3) - 1) * 0.2 for s in range(NUM_STATIONS)], dtype=float)

_TOOL_BASE_R = np.asarray([0.8, 0.6, 0.5], dtype=float)
_TOOL_BASE_D = np.asarray([0.06, 0.08, 0.10], dtype=float)
_TOOL_BASE_E = np.asarray([0.02, -0.01, 0.03], dtype=float)

QUALITY_R_CACHE = np.zeros((NUM_TASKS, NUM_STATIONS, NUM_TOOL_TYPES), dtype=float)
QUALITY_D_CACHE = np.zeros((NUM_TASKS, NUM_STATIONS, NUM_TOOL_TYPES), dtype=float)
QUALITY_E_CACHE = np.zeros((NUM_TASKS, NUM_STATIONS, NUM_TOOL_TYPES), dtype=float)

for _task in range(NUM_TASKS):
    for _station in range(NUM_STATIONS):
        _scale = TASK_SCALE_CACHE[_task] * STATION_ADJUST_CACHE[_station]
        QUALITY_R_CACHE[_task, _station, :] = _TOOL_BASE_R * _scale
        QUALITY_D_CACHE[_task, _station, :] = np.clip(_TOOL_BASE_D * _scale, 0.0, 0.5)
        QUALITY_E_CACHE[_task, _station, :] = _TOOL_BASE_E * _scale


# 成本统计
COST_STATS = {}

def reset_cost_stats():
    global COST_STATS
    COST_STATS = {
        "n_obj_eval_total": 0,
        "n_local_search_calls": 0,
        "time_local_search_sec": 0.0,
        "runtime_sec": 0.0,
    }


def get_cost_stats_snapshot():
    stats = dict(COST_STATS)
    if stats["runtime_sec"] > 0:
        stats["local_search_time_ratio_pct"] = (stats["time_local_search_sec"] / stats["runtime_sec"] * 100.0)
    else:
        stats["local_search_time_ratio_pct"] = 0.0
    return stats


def get_environment_info():
    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
    }


# 质量损失计算模块
def simple_quality_model(task: int, station: int, tool: int):
    if 0 <= int(tool) <= 2:
        t = int(task)
        s = int(station)
        u = int(tool)
        return (
            float(QUALITY_R_CACHE[t, s, u]),
            float(QUALITY_D_CACHE[t, s, u]),
            float(QUALITY_E_CACHE[t, s, u]),
        )

    rb, db, eb = (0.7, 0.08, 0.0)
    task_scale = 1.0 + (TASK_TIMES[task] / (MAX_TASK_TIME + EPS)) * 0.2
    station_adjust = 1.0 + ((station % 3) - 1) * 0.2
    roughness = rb * task_scale * station_adjust
    defect_rate = min(max(db * task_scale * station_adjust, 0.0), 0.5)
    expansion = eb * task_scale * station_adjust
    return roughness, defect_rate, expansion


def compute_average_quality_indicators(solution):
    station_assignment, _, tool_assignment = solution
    if NUM_TASKS == 0:
        return 0.0, 0.0, 0.0

    stations = np.asarray(station_assignment, dtype=int)
    tools = np.asarray(tool_assignment, dtype=int)
    task_idx = np.arange(NUM_TASKS, dtype=int)

    total_roughness = float(np.sum(QUALITY_R_CACHE[task_idx, stations, tools]))
    total_defect_rate = float(np.sum(QUALITY_D_CACHE[task_idx, stations, tools]))
    total_expansion = float(np.sum(QUALITY_E_CACHE[task_idx, stations, tools]))

    avg_roughness = total_roughness / NUM_TASKS
    avg_defect_rate = total_defect_rate / NUM_TASKS
    avg_expansion = total_expansion / NUM_TASKS
    return avg_roughness, avg_defect_rate, avg_expansion


def normalize_quality_components(avg_roughness, avg_defect_rate, avg_expansion, params=None):
    if params is None:
        params = BASE_QUALITY_PARAMS

    R_min = params["R_min"]
    R_max = params["R_max"]
    D_max = params["D_max"]
    E_tol = params["E_tol"]
    E_target = params["E_target"]
    w_plus = params["w_plus"]
    w_minus = params["w_minus"]

    # 1) 粗糙度归一化
    if R_max > R_min:
        roughness_norm = (avg_roughness - R_min) / (R_MAX - R_MIN)
        roughness_norm = max(0.0, min(1.0, roughness_norm))
    else:
        roughness_norm = 0.0

    # 2) 缺陷率归一化（比例值 / 比例上限）
    if D_max > 0:
        defect_norm = avg_defect_rate / D_max
        defect_norm = max(0.0, min(1.0, defect_norm))
    else:
        defect_norm = 0.0

    # 3) 膨胀偏差归一化（不对称惩罚）
    delta_e = avg_expansion - E_target
    if E_tol > 0:
        if delta_e >= 0:
            expansion_norm = w_plus * (delta_e / E_tol)
        else:
            expansion_norm = w_minus * (abs(delta_e) / E_tol)
    else:
        expansion_norm = 0.0

    return roughness_norm, defect_norm, expansion_norm


def calculate_quality_loss(solution, weights=None, params=None, return_components=False) -> Union[
    Dict[str, float], float]:
    if weights is None:
        weights = BASE_WEIGHTS
    if params is None:
        params = BASE_QUALITY_PARAMS

    weights = np.asarray(weights, dtype=float)
    weights = weights / (weights.sum() + EPS)

    avg_roughness, avg_defect_rate, avg_expansion = compute_average_quality_indicators(solution)
    roughness_norm, defect_norm, expansion_norm = normalize_quality_components(
        avg_roughness, avg_defect_rate, avg_expansion, params
    )

    qloss = (
        weights[0] * roughness_norm +
        weights[1] * defect_norm +
        weights[2] * expansion_norm
    )

    if return_components:
        return {
            "quality_loss": float(qloss),
            "avg_roughness": float(avg_roughness),
            "avg_defect_rate": float(avg_defect_rate),
            "avg_expansion": float(avg_expansion),
            "roughness_norm": float(roughness_norm),
            "defect_norm": float(defect_norm),
            "expansion_norm": float(expansion_norm),
        }

    return float(qloss)


def calculate_quality_loss_with_params(solution, params):
    return calculate_quality_loss(solution, weights=BASE_WEIGHTS, params=params)


# 多目标评估模块
def evaluate_welding_objectives_with_penalty(solution):
    """评估三目标值（含约束违反惩罚），并统计总评估次数"""
    COST_STATS["n_obj_eval_total"] += 1
    station_assignment, station_sequences, tool_assignment = solution
    penalty = 0.0

    pos_in_station = np.full(NUM_TASKS, -1, dtype=int)
    for s in range(NUM_STATIONS):
        seq = station_sequences[s]
        for idx, task in enumerate(seq):
            pos_in_station[task] = idx

    for pre, post in PRECEDENCE_CONSTRAINTS:
        if station_assignment[pre] > station_assignment[post]:
            penalty += 15.0
        elif station_assignment[pre] == station_assignment[post]:
            pre_pos = pos_in_station[pre]
            post_pos = pos_in_station[post]
            if pre_pos < 0 or post_pos < 0:
                penalty += 5.0
            elif pre_pos > post_pos:
                penalty += 10.0

    station_times = np.zeros(NUM_STATIONS, dtype=float)
    for s in range(NUM_STATIONS):
        seq = station_sequences[s]
        if not seq:
            continue
        first_task = seq[0]
        station_times[s] += TASK_TIMES[first_task]
        current_tool = int(tool_assignment[first_task])
        for idx in range(1, len(seq)):
            task = seq[idx]
            prev_tool = current_tool
            current_tool = int(tool_assignment[task])
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


# 8. 支配关系与档案管理
def dominates_eps(a, b):
    ct_dominated = a[0] <= b[0] + GLOBAL_EPS_CT
    std_dominated = a[1] <= b[1] + GLOBAL_EPS_STD
    qloss_dominated = a[2] < b[2]
    all_dominated = ct_dominated and std_dominated and qloss_dominated
    ct_strict = a[0] < b[0] - GLOBAL_EPS_CT
    std_strict = a[1] < b[1] - GLOBAL_EPS_STD
    qloss_strict = a[2] < b[2]
    any_strict = ct_strict or std_strict or qloss_strict
    return all_dominated and any_strict


def dominates_standard(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.all(a <= b + EPS) and np.any(a < b - EPS)


def calculate_crowding_distance(front):
    n = len(front)
    if n == 0:
        return []
    objectives = np.array([obj for _, obj in front], dtype=float)
    crowding_distances = np.zeros(n)
    for m in range(objectives.shape[1]):
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


def update_archive(epsilon_archive, cand_solution, cand_obj):
    for _, obj in epsilon_archive:
        if dominates_eps(obj, cand_obj):
            return epsilon_archive
    new_archive = [(sol, obj) for sol, obj in epsilon_archive if not dominates_eps(cand_obj, obj)]
    new_archive.append((cand_solution, cand_obj))
    if len(new_archive) > MAX_ARCHIVE_SIZE:
        crowding_distances = calculate_crowding_distance(new_archive)
        sorted_indices = np.argsort(crowding_distances)[::-1]
        new_archive = [new_archive[i] for i in sorted_indices[:MAX_ARCHIVE_SIZE]]
    return new_archive


def strict_pareto_filter(solution_obj_list):
    if not solution_obj_list:
        return []
    filtered = []
    n = len(solution_obj_list)
    for i in range(n):
        sol_i, obj_i = solution_obj_list[i]
        dominated = False
        duplicate = False
        for j in range(n):
            if i == j:
                continue
            _, obj_j = solution_obj_list[j]
            if dominates_standard(obj_j, obj_i):
                dominated = True
                break
            if np.allclose(np.asarray(obj_i, dtype=float), np.asarray(obj_j, dtype=float), atol=1e-12, rtol=0.0):
                if j < i:
                    duplicate = True
                    break
        if not dominated and not duplicate:
            filtered.append((sol_i, obj_i))
    return filtered


# 9. 代表解/精英解选择
def select_representative_solutions(front, w=(0.6, 0.2, 0.2)):
    reps = []
    if not front:
        return reps
    w = np.asarray(w, dtype=float)
    w = w / (w.sum() + EPS)
    objs = np.array([obj for _, obj in front], dtype=float)
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    ranges = np.maximum(maxs - mins, EPS)
    norm_objs = (objs - mins) / ranges
    d = np.sqrt(np.sum(w * (norm_objs * norm_objs), axis=1))
    idx_ideal = int(np.argmin(d))
    reps.append(("Ideal-point", front[idx_ideal]))
    return reps


def select_topk_elite_archive_solutions(epsilon_archive, top_k=ELITE_TOP_K, w=IDEAL_POINT_WEIGHTS):
    if not epsilon_archive:
        return []
    w = np.asarray(w, dtype=float)
    w = w / (w.sum() + EPS)
    objs = np.array([obj for _, obj in epsilon_archive], dtype=float)
    mins = objs.min(axis=0)
    maxs = objs.max(axis=0)
    ranges = np.maximum(maxs - mins, EPS)
    norm_objs = (objs - mins) / ranges
    d = np.sqrt(np.sum(w * (norm_objs * norm_objs), axis=1))
    elite_indices = np.argsort(d)[:min(top_k, len(epsilon_archive))]
    return [epsilon_archive[i] for i in elite_indices]


def rebuild_epsilon_archive_from_candidates(candidates):
    archive = []
    for sol, obj in candidates:
        archive = update_archive(archive, sol, obj)
    return archive


def majority_beats(obj_a, obj_b, k=2):
    obj_a = np.asarray(obj_a, dtype=float)
    obj_b = np.asarray(obj_b, dtype=float)
    better_cnt = int(np.sum(obj_a < obj_b - EPS))
    return better_cnt >= k


def select_final_solution_majority(final_pareto_front):
    if not final_pareto_front:
        return None, None, {"wins": [], "losses": [], "scores": []}
    objs = [obj for _, obj in final_pareto_front]
    n = len(objs)
    wins = np.zeros(n, dtype=int)
    losses = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if majority_beats(objs[i], objs[j], k=2):
                wins[i] += 1
            if majority_beats(objs[j], objs[i], k=2):
                losses[i] += 1
    scores = wins - losses
    best_score = np.max(scores)
    cand_idx = np.where(scores == best_score)[0].tolist()

    def tie_key(i):
        ct, std, ql = objs[i]
        return (ct, std, ql)

    best_i = min(cand_idx, key=tie_key)
    best_sol, best_obj = final_pareto_front[best_i]
    stats = {"best_score": int(best_score), "num_solutions": n}
    return best_sol, best_obj, stats


# =============================================================================
# 10. 量子观测与更新
# =============================================================================
def observe_individual(Q_j, n_obs):
    candidates = []
    for _ in range(n_obs):
        observed_bits = []
        ptr = 0
        station_assignment = np.zeros(NUM_TASKS, dtype=int)
        tool_assignment = np.zeros(NUM_TASKS, dtype=int)
        rand_keys = np.zeros(NUM_TASKS)

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
            if i != j and dominates_eps(penalized_obj_j, penalized_obj_i):
                cnt += 1
        dominated_counts.append(cnt)
    best_index = int(np.argmin(dominated_counts))
    return (
        candidates[best_index][0],
        candidates[best_index][1],
        candidates[best_index][2],
        candidates[best_index][3],
    )


def update_Q(Q, guided_archive, X_obs, t, max_iter):
    pop_size, num_qubits, _ = Q.shape
    new_Q = np.copy(Q)
    valid_size = len(guided_archive)
    if valid_size == 0:
        return new_Q
    progress = t / max_iter

    reps = select_representative_solutions(guided_archive)
    if reps:
        p_random = 0.45 * (1 - progress) + 0.15
        p_ideal = 1.0 - p_random
        choice = np.random.choice(["ideal", "random"], p=[p_ideal, p_random])
        if choice == "ideal":
            ref_solution = reps[0][1][0]
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
    ref_pos = np.full(NUM_TASKS, -1, dtype=int)
    ref_len = np.zeros(NUM_STATIONS, dtype=int)
    for s in range(NUM_STATIONS):
        seq = station_sequences[s]
        ref_len[s] = len(seq)
        for idx, task in enumerate(seq):
            ref_pos[task] = idx
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
        rk = ref_pos[task]
        if rk >= 0 and ref_len[s] > 0:
            ref_bits.append(1 if rk >= ref_len[s] / 2 else 0)
        else:
            ref_bits.append(0)
    ref_bits = np.array(ref_bits, dtype=int)

    base_angle = 0.03 * np.pi * np.exp(-t / (max_iter / 2))
    archive_size = len(guided_archive)
    p_mut_min, p_mut_max = 0.006, 0.06
    size_factor = 1.0 / (1.0 + np.log1p(max(archive_size, 1)))
    p_mut = p_mut_min + (p_mut_max - p_mut_min) * (1 - progress) * size_factor
    phi_max = (0.04 * np.pi) * (1 - progress) + 0.02 * np.pi
    p_reinit, p_flip = 0.4 * p_mut, 0.4 * p_mut

    update_indices = np.random.choice(pop_size, size=valid_size, replace=False)
    for idx in range(valid_size):
        j = update_indices[idx]
        for i in range(num_qubits):
            alpha, beta = Q[j, i]
            xi = X_obs[idx, i]
            ri = ref_bits[i]
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
            p1 = float(abs(new_beta) ** 2)
            polarization = abs(p1 - 0.5)
            boost = 1.0 + 1.5 * max(0.0, polarization - 0.25)
            if np.random.rand() < p_mut * boost:
                r = np.random.rand()
                if r < p_reinit:
                    new_alpha = new_beta = 1 / np.sqrt(2)
                elif r < p_reinit + p_flip:
                    new_alpha, new_beta = new_beta, new_alpha
                    jitter = np.random.uniform(-0.01 * np.pi, 0.01 * np.pi)
                    c2, s2 = np.cos(jitter), np.sin(jitter)
                    new_alpha, new_beta = c2 * new_alpha - s2 * new_beta, s2 * new_alpha + c2 * new_beta
                else:
                    delta = np.random.uniform(-phi_max, phi_max)
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


# 11. 局部 QLoss 改进
def local_qloss_improvement(solution, obj_values):
    COST_STATS["n_local_search_calls"] += 1
    t0 = time.perf_counter()
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
            ct_ok = test_ct <= ct + LOCAL_EPS_CT
            std_ok = test_std <= load_std + LOCAL_EPS_STD
            qloss_improved = test_qloss < qloss - EPS
            if ct_ok and std_ok and qloss_improved:
                new_tool_assignment = test_tool_assignment
                ct, load_std, qloss = test_ct, test_std, test_qloss
                improved = True
                break

    COST_STATS["time_local_search_sec"] += time.perf_counter() - t0
    if improved:
        improved_solution = (station_assignment, station_sequences, new_tool_assignment)
        improved_obj = (ct, load_std, qloss)
        return improved_solution, improved_obj
    return solution, obj_values


def has_empty_station(solution) -> bool:
    _, station_sequences, _ = solution
    return any(len(seq) == 0 for seq in station_sequences)


def compute_station_times(solution):
    _, station_sequences, tool_assignment = solution
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


# 12. IQEA 主循环
def quantum_evolutionary_optimization(pop_size=30, max_iter=100, n_obs_base=3, verbose=True):
    reset_cost_stats()
    t_run_start = time.perf_counter()

    num_qubits = NUM_TASKS * (NUM_STATIONS + NUM_TOOL_TYPES + 1)
    Q = np.zeros((pop_size, num_qubits, 2))
    Q[:, :, 0] = Q[:, :, 1] = 1 / np.sqrt(2)

    epsilon_archive = []
    history = {'best_ct': [], 'best_load_std': [], 'best_qloss': [], 'archive_size': []}

    for t in range(max_iter):
        solutions = []
        original_objs = []
        penalized_objs = []
        X_obs = []
        progress = t / max_iter
        n_obs = n_obs_base + int(2 * progress) if progress < 1.0 else n_obs_base + 2

        for j in range(pop_size):
            sol, obs_bits, orig_obj, penalized_obj = observe_individual(Q[j], n_obs)
            if not has_empty_station(sol):
                solutions.append(sol)
                original_objs.append(orig_obj)
                penalized_objs.append(penalized_obj)
                X_obs.append(obs_bits)

        X_obs = np.array(X_obs) if X_obs else np.empty((0, num_qubits))

        for sol, orig_obj in zip(solutions, original_objs):
            if not np.any(np.isinf(orig_obj)):
                epsilon_archive = update_archive(epsilon_archive, sol, orig_obj)

        if progress > 0.5 and epsilon_archive:
            elite_solutions = select_topk_elite_archive_solutions(
                epsilon_archive, top_k=ELITE_TOP_K, w=IDEAL_POINT_WEIGHTS
            )
            elite_ids = {id(sol) for sol, _ in elite_solutions}
            merged_candidates = []
            for sol, obj in epsilon_archive:
                if id(sol) in elite_ids:
                    improved_sol, improved_obj = local_qloss_improvement(sol, obj)
                    merged_candidates.append((improved_sol, improved_obj))
                else:
                    merged_candidates.append((sol, obj))
            epsilon_archive = rebuild_epsilon_archive_from_candidates(merged_candidates)

        if epsilon_archive:
            archive_objs = np.array([obj for _, obj in epsilon_archive], dtype=float)
            history['best_ct'].append(archive_objs[:, 0].min())
            history['best_load_std'].append(archive_objs[:, 1].min())
            history['best_qloss'].append(archive_objs[:, 2].min())
        else:
            history['best_ct'].append(np.inf)
            history['best_load_std'].append(np.inf)
            history['best_qloss'].append(np.inf)
        history['archive_size'].append(len(epsilon_archive))

        guided_archive = [(sol, pen_obj) for sol, pen_obj in zip(solutions, penalized_objs)]
        Q = update_Q(Q, guided_archive, X_obs, t, max_iter)

        if verbose and (t % 10 == 0):
            print(
                f"IQEA Iter {t:3d}/{max_iter:3d} | "
                f"Epsilon-Archive Size: {len(epsilon_archive):2d} | "
                f"Valid Solutions: {len(guided_archive):2d} | "
                f"Best CT: {history['best_ct'][-1]:6.2f} | "
                f"Best LoadSTD: {history['best_load_std'][-1]:6.2f} | "
                f"Best QLoss: {history['best_qloss'][-1]:6.4f}"
            )

    final_pareto_front = strict_pareto_filter(epsilon_archive)
    COST_STATS["runtime_sec"] = time.perf_counter() - t_run_start
    cost_stats = get_cost_stats_snapshot()
    return epsilon_archive, final_pareto_front, history, cost_stats


# 13. 敏感性分析工具函数
def rank_vector_from_scores(scores):
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=int)
    ranks[order] = np.arange(len(scores))
    return ranks, order


def topk_overlap_ratio(order_a, order_b, k=5):
    k = min(k, len(order_a), len(order_b))
    if k <= 0:
        return 0.0
    set_a = set(order_a[:k].tolist() if hasattr(order_a, "tolist") else order_a[:k])
    set_b = set(order_b[:k].tolist() if hasattr(order_b, "tolist") else order_b[:k])
    return len(set_a & set_b) / float(k)


def safe_spearman(rank_vec_a, rank_vec_b):
    if len(rank_vec_a) <= 1 or len(rank_vec_b) <= 1:
        return 1.0
    corr, _ = spearmanr(rank_vec_a, rank_vec_b)
    return float(corr) if np.isfinite(corr) else 0.0


# =============================================================================
# 14. 权重敏感性分析（基于标准 Pareto 前沿）
# =============================================================================
def weight_sensitivity_analysis(final_pareto_front, base_weights=None, n_samples=2000, top_k=5, random_state=2025):
    if base_weights is None:
        base_weights = BASE_WEIGHTS.tolist()

    if not final_pareto_front:
        print("标准 Pareto 前沿为空，无法进行权重敏感性分析")
        return {}

    print(f"开始权重敏感性分析，使用 {len(final_pareto_front)} 个标准Pareto解，采样 {n_samples} 组权重...")

    solutions = [sol for sol, _ in final_pareto_front]
    n_sol = len(solutions)

    base_scores = [calculate_quality_loss(sol, weights=base_weights, params=BASE_QUALITY_PARAMS) for sol in solutions]
    base_rank_vec, base_order = rank_vector_from_scores(base_scores)

    rng = np.random.default_rng(random_state)
    weights_samples = rng.dirichlet(np.ones(3), size=n_samples)

    spearman_corrs = np.zeros(n_samples, dtype=float)
    topk_overlaps = np.zeros(n_samples, dtype=float)

    for i, weights in enumerate(weights_samples):
        if i % 200 == 0:
            print(f"  进度: {i + 1}/{n_samples} ({(i + 1) / n_samples * 100:.1f}%)")

        current_scores = [calculate_quality_loss(sol, weights=weights, params=BASE_QUALITY_PARAMS) for sol in solutions]
        current_rank_vec, current_order = rank_vector_from_scores(current_scores)

        topk_overlaps[i] = topk_overlap_ratio(base_order, current_order, k=top_k)
        spearman_corrs[i] = safe_spearman(base_rank_vec, current_rank_vec)

    results = {
        'n_solutions': n_sol,
        'n_samples': int(n_samples),
        'base_weights': [float(x) for x in base_weights],
        'top_k': int(top_k),

        'topk_overlap_mean': float(np.mean(topk_overlaps) * 100),
        'topk_overlap_median': float(np.median(topk_overlaps) * 100),
        'topk_overlap_p10_p90': [
            float(np.percentile(topk_overlaps, 10) * 100),
            float(np.percentile(topk_overlaps, 90) * 100)
        ],

        'spearman_median': float(np.median(spearman_corrs)),
        'spearman_mean': float(np.mean(spearman_corrs)),
        'spearman_std': float(np.std(spearman_corrs)),
        'spearman_p10_p90': [
            float(np.percentile(spearman_corrs, 10)),
            float(np.percentile(spearman_corrs, 90))
        ],

        'raw': {
            'topk_overlaps': topk_overlaps.tolist(),
            'spearman_corrs': spearman_corrs.tolist(),
        }
    }
    return results


# 15. 参数敏感性分析
def build_parameter_scenarios():
    param_variations = {
        'R_range': [(0.4, 2.5), (0.5, 2.0), (0.6, 1.8)],
        'D_max': [0.08, 0.10, 0.12],
        'E_tol': [0.10, 0.15, 0.20],
    }
    return list(itertools.product(
        param_variations['R_range'],
        param_variations['D_max'],
        param_variations['E_tol']
    ))


def select_representative_combinations(all_combinations, n_scenarios):
    total = len(all_combinations)
    if n_scenarios is None or n_scenarios >= total:
        return all_combinations

    indices = np.linspace(0, total - 1, n_scenarios, dtype=int)
    selected = [all_combinations[i] for i in sorted(set(indices.tolist()))]

    if len(selected) < n_scenarios:
        used = set(indices.tolist())
        for i in range(total):
            if i not in used:
                selected.append(all_combinations[i])
                if len(selected) == n_scenarios:
                    break
    return selected


def parameter_sensitivity_analysis(final_pareto_front, n_scenarios=27, top_k=5):
    if not final_pareto_front:
        print("标准 Pareto 前沿为空，无法进行参数敏感性分析")
        return {}

    all_param_combinations = build_parameter_scenarios()
    selected_combinations = select_representative_combinations(all_param_combinations, n_scenarios)

    solutions = [sol for sol, _ in final_pareto_front]
    n_sol = len(solutions)

    base_params = dict(BASE_QUALITY_PARAMS)

    print(f"开始参数范围敏感性分析，测试 {len(selected_combinations)} 个场景...")

    base_scores = [calculate_quality_loss_with_params(sol, base_params) for sol in solutions]
    base_rank_vec, base_order = rank_vector_from_scores(base_scores)

    details = []
    for i, (R_range, D_max_s, E_tol_s) in enumerate(selected_combinations):
        if i % 3 == 0:
            print(f"  进度: {i + 1}/{len(selected_combinations)} ({(i + 1) / len(selected_combinations) * 100:.1f}%)")

        current_params = dict(base_params)
        current_params['R_min'], current_params['R_max'] = R_range
        current_params['D_max'] = D_max_s
        current_params['E_tol'] = E_tol_s

        current_scores = [calculate_quality_loss_with_params(sol, current_params) for sol in solutions]
        current_rank_vec, current_order = rank_vector_from_scores(current_scores)

        overlap = topk_overlap_ratio(base_order, current_order, k=top_k)
        corr = safe_spearman(base_rank_vec, current_rank_vec)

        details.append({
            'params': {
                'R_min': float(current_params['R_min']),
                'R_max': float(current_params['R_max']),
                'D_max': float(current_params['D_max']),
                'E_tol': float(current_params['E_tol']),
                'E_target': float(current_params['E_target']),
                'w_plus': float(current_params['w_plus']),
                'w_minus': float(current_params['w_minus']),
            },
            'topk_overlap': float(overlap),
            'spearman_corr': float(corr),
        })

    topk_overlaps = np.array([r['topk_overlap'] for r in details], dtype=float)
    spearman_corrs = np.array([r['spearman_corr'] for r in details], dtype=float)

    summary = {
        'n_scenarios': len(details),
        'n_solutions': n_sol,
        'top_k': int(top_k),

        'topk_overlap_mean': float(np.mean(topk_overlaps) * 100),
        'topk_overlap_std': float(np.std(topk_overlaps) * 100),

        'spearman_mean': float(np.mean(spearman_corrs)),
        'spearman_std': float(np.std(spearman_corrs)),

        'details': details
    }
    return summary


# 16. 质量分量诊断
def diagnose_quality_components(final_pareto_front):
    if not final_pareto_front:
        return {}

    component_rows = []
    for sol, _ in final_pareto_front:
        info = calculate_quality_loss(
            sol,
            weights=BASE_WEIGHTS,
            params=BASE_QUALITY_PARAMS,
            return_components=True
        )
        component_rows.append(info)

    avg_roughness = np.array([x["avg_roughness"] for x in component_rows], dtype=float)
    avg_defect_rate = np.array([x["avg_defect_rate"] for x in component_rows], dtype=float)
    avg_expansion = np.array([x["avg_expansion"] for x in component_rows], dtype=float)
    roughness_norm = np.array([x["roughness_norm"] for x in component_rows], dtype=float)
    defect_norm = np.array([x["defect_norm"] for x in component_rows], dtype=float)
    expansion_norm = np.array([x["expansion_norm"] for x in component_rows], dtype=float)

    positive_expansion_ratio = float(np.mean(avg_expansion > TARGET_EXPANSION) * 100)

    return {
        "n_solutions": len(final_pareto_front),
        "avg_roughness_mean": float(np.mean(avg_roughness)),
        "avg_roughness_range": [float(np.min(avg_roughness)), float(np.max(avg_roughness))],

        "avg_defect_rate_mean": float(np.mean(avg_defect_rate)),
        "avg_defect_rate_range": [float(np.min(avg_defect_rate)), float(np.max(avg_defect_rate))],

        "avg_expansion_mean": float(np.mean(avg_expansion)),
        "avg_expansion_range": [float(np.min(avg_expansion)), float(np.max(avg_expansion))],
        "positive_expansion_ratio_pct": positive_expansion_ratio,

        "roughness_norm_mean": float(np.mean(roughness_norm)),
        "defect_norm_mean": float(np.mean(defect_norm)),
        "expansion_norm_mean": float(np.mean(expansion_norm)),
    }


def print_quality_diagnostics(diag):
    if not diag:
        print("无质量诊断结果")
        return

    print("\n" + "=" * 80)
    print("Pareto解质量分量诊断")
    print("=" * 80)
    print(f"解数量: {diag['n_solutions']}")
    print(f"平均粗糙度: {diag['avg_roughness_mean']:.4f} | 范围: [{diag['avg_roughness_range'][0]:.4f}, {diag['avg_roughness_range'][1]:.4f}]")
    print(f"平均缺陷率: {diag['avg_defect_rate_mean']:.4f} | 范围: [{diag['avg_defect_rate_range'][0]:.4f}, {diag['avg_defect_rate_range'][1]:.4f}]")
    print(f"平均膨胀量: {diag['avg_expansion_mean']:.4f} | 范围: [{diag['avg_expansion_range'][0]:.4f}, {diag['avg_expansion_range'][1]:.4f}]")
    print(f"膨胀量超过目标值 ({TARGET_EXPANSION:.3f}) 的解占比: {diag['positive_expansion_ratio_pct']:.1f}%")
    print(f"平均归一化粗糙度分量: {diag['roughness_norm_mean']:.4f}")
    print(f"平均归一化缺陷率分量: {diag['defect_norm_mean']:.4f}")
    print(f"平均归一化膨胀分量: {diag['expansion_norm_mean']:.4f}")
    print("=" * 80)


def print_pareto_summary(final_pareto_front, top_k=20):
    if not final_pareto_front:
        print("\n最终标准Pareto前沿为空，无前沿解可展示")
        return
    sorted_front = sorted(final_pareto_front, key=lambda x: x[1][0])
    total = len(sorted_front)
    print("\n" + "=" * 80)
    print(f"IQEA最终标准Pareto前沿解汇总（共{total}个非支配解，展示前{min(top_k, total)}个）")
    print("=" * 80)
    print(f"{'序号':<4} {'生产节拍CT(秒)':<12} {'负载标准差(秒)':<12} {'质量损失QLoss':<12} {'工位负载分布(秒)':<30}")
    print("-" * 80)
    for i, (sol, obj) in enumerate(sorted_front[:top_k], start=1):
        ct, load_std, qloss = obj
        station_times = compute_station_times(sol)
        load_dist = "[" + ", ".join([f"{t:.1f}" for t in station_times]) + "]"
        print(f"{i:<4} {ct:<12.2f} {load_std:<12.2f} {qloss:<12.4f} {load_dist:<30}")
    all_objs = np.array([obj for _, obj in sorted_front], dtype=float)
    print("-" * 80)
    print("统计信息：")
    print(f"  生产节拍CT：最小值={all_objs[:, 0].min():.2f} | 平均值={all_objs[:, 0].mean():.2f} | 最大值={all_objs[:, 0].max():.2f}")
    print(f"  负载标准差：最小值={all_objs[:, 1].min():.2f} | 平均值={all_objs[:, 1].mean():.2f} | 最大值={all_objs[:, 1].max():.2f}")
    print(f"  质量损失：最小值={all_objs[:, 2].min():.4f} | 平均值={all_objs[:, 2].mean():.4f} | 最大值={all_objs[:, 2].max():.4f}")
    print("=" * 80)


def print_sensitivity_results(weight_results, param_results=None):
    print("\n" + "=" * 80)
    print("权重敏感性分析结果（基于标准Pareto前沿，rank-based）")
    print("=" * 80)

    if weight_results:
        print("分析设置:")
        print(f"  解数量: {weight_results.get('n_solutions', 'N/A')}")
        print(f"  权重样本数: {weight_results.get('n_samples', 'N/A')}")
        bw = weight_results.get('base_weights', [0.2, 0.5, 0.3])
        print(f"  基线权重: α={bw[0]}, β={bw[1]}, γ={bw[2]}")
        print(f"  Top-k: {weight_results.get('top_k', 5)}")

        print("\n关键结果:")
        print("  1) 稳定性（Top-k重合率）:")
        print(f"     Top-{weight_results['top_k']}重合率（均值）: {weight_results['topk_overlap_mean']:.1f}%")
        print(f"     Top-{weight_results['top_k']}重合率（中位数）: {weight_results['topk_overlap_median']:.1f}%")
        p10, p90 = weight_results['topk_overlap_p10_p90']
        print(f"     P10-P90范围: [{p10:.1f}%, {p90:.1f}%]")

        print("  2) 排序一致性（Spearman，基于名次向量）:")
        print(f"     中位数: {weight_results['spearman_median']:.3f}")
        print(f"     平均值: {weight_results['spearman_mean']:.3f} ± {weight_results['spearman_std']:.3f}")
        sp10, sp90 = weight_results['spearman_p10_p90']
        print(f"     P10-P90范围: [{sp10:.3f}, {sp90:.3f}]")
    else:
        print("权重敏感性分析结果为空")

    if param_results:
        print("\n" + "=" * 80)
        print("参数范围敏感性分析结果（基于标准Pareto前沿，rank-based）")
        print("=" * 80)

        print("分析设置:")
        print(f"  测试场景数: {param_results.get('n_scenarios', 'N/A')}")
        print(f"  Top-k: {param_results.get('top_k', 5)}")

        print("\n关键结果:")
        print("  1) 稳定性（Top-k重合率）:")
        print(f"     Top-{param_results['top_k']}重合率（均值）: {param_results['topk_overlap_mean']:.1f}% ± {param_results['topk_overlap_std']:.1f}%")

        print("  2) 排序一致性（Spearman，基于名次向量）:")
        print(f"     平均值: {param_results['spearman_mean']:.3f} ± {param_results['spearman_std']:.3f}")

    print("\n" + "=" * 80)
    print("结论:")
    if weight_results:
        if weight_results.get('topk_overlap_mean', 0) > 70:
            print("✓ 权重扰动下Top-k重合率较高，方案集合具有一定稳定性")
        if weight_results.get('spearman_median', 0) > 0.8:
            print("✓ Spearman（名次向量）中位数较高，排序一致性强")
    if param_results and param_results.get('topk_overlap_mean', 0) > 70:
        print("✓ 归一化参数扰动下Top-k重合率较高，排序结构整体鲁棒")
    print("注：优化、最终输出、敏感性分析均基于与 safe_speedup 版一致的标准Pareto前沿流程。")
    print("=" * 80)


def plot_optimization_history(history):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history['best_ct'], linewidth=2, marker='o', markersize=3)
    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].set_ylabel("CT（s）")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(history['best_load_std'], linewidth=2, marker='s', markersize=3)
    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylabel("LoadSTD（s）")
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(history['best_qloss'], linewidth=2, marker='^', markersize=3)
    axes[1, 0].set_title("每代内部档案最优质量损失(归一化QLoss)")
    axes[1, 0].set_xlabel("迭代次数")
    axes[1, 0].set_ylabel("质量损失")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(history['archive_size'], linewidth=2, marker='d', markersize=3)
    axes[1, 1].set_title("内部ε-档案大小")
    axes[1, 1].set_xlabel("迭代次数")
    axes[1, 1].set_ylabel("解数量")
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_3d_pareto_front(final_pareto_front):
    if not final_pareto_front:
        print("最终标准Pareto前沿为空，无法绘制3D前沿图")
        return
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ct_vals = [obj[0] for _, obj in final_pareto_front]
    load_std_vals = [obj[1] for _, obj in final_pareto_front]
    qloss_vals = [obj[2] for _, obj in final_pareto_front]
    sc = ax.scatter(ct_vals, load_std_vals, qloss_vals, c=qloss_vals, cmap='viridis', s=60, alpha=0.8, edgecolors='black', linewidths=0.5)
    ax.set_xlabel('生产节拍CT（秒）', fontsize=11)
    ax.set_ylabel('负载标准差LoadSTD（秒）', fontsize=11)
    ax.set_zlabel('质量损失QLoss', fontsize=11)
    ax.set_title('IQEA最终标准Pareto前沿（3D视图）', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label('质量损失QLoss', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_2d_pareto_projections(final_pareto_front):
    if not final_pareto_front:
        print("最终标准Pareto前沿为空，无法绘制2D投影图")
        return
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("IQEA最终标准Pareto前沿二维投影", fontsize=14, fontweight='bold')
    ct_vals = np.array([obj[0] for _, obj in final_pareto_front], dtype=float)
    load_std_vals = np.array([obj[1] for _, obj in final_pareto_front], dtype=float)
    qloss_vals = np.array([obj[2] for _, obj in final_pareto_front], dtype=float)
    axes[0].scatter(ct_vals, load_std_vals, s=60, alpha=0.8, edgecolors='black', linewidths=0.5)
    axes[0].set_xlabel('生产节拍CT（秒）', fontsize=11)
    axes[0].set_ylabel('负载标准差LoadSTD（秒）', fontsize=11)
    axes[0].set_title('CT vs LoadSTD', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(ct_vals, qloss_vals, s=60, alpha=0.8, edgecolors='black', linewidths=0.5)
    axes[1].set_xlabel('生产节拍CT（秒）', fontsize=11)
    axes[1].set_ylabel('质量损失QLoss', fontsize=11)
    axes[1].set_title('CT vs QLoss', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[2].scatter(load_std_vals, qloss_vals, s=60, alpha=0.8, edgecolors='black', linewidths=0.5)
    axes[2].set_xlabel('负载标准差LoadSTD（秒）', fontsize=11)
    axes[2].set_ylabel('质量损失QLoss', fontsize=11)
    axes[2].set_title('LoadSTD vs QLoss', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sensitivity_results(weight_results, param_results=None):
    if not weight_results or 'raw' not in weight_results:
        print("无权重敏感性分析结果可可视化（缺少raw数据）")
        return

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    topk = weight_results.get('top_k', 5)
    topk_overlaps = np.array(weight_results['raw']['topk_overlaps'], dtype=float) * 100.0
    spearman_data = np.array(weight_results['raw']['spearman_corrs'], dtype=float)

    if param_results:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        ax1, ax2, ax3 = axes
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1, ax2 = axes
        ax3 = None

    ax1.bar([f'Top-{topk}'], [weight_results['topk_overlap_mean']], color='skyblue')
    ax1.axhline(y=75, color='r', linestyle='--', alpha=0.5, label='参考阈值 (75%)')
    ax1.set_ylabel('重合率 (%)')
    ax1.set_title('方案集合稳定性（Top-k重合率均值）')
    ax1.set_ylim([0, 100])
    ax1.legend()

    ax2.hist(spearman_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(x=weight_results['spearman_median'], color='r', linestyle='-', linewidth=2,
                label=f'中位数: {weight_results["spearman_median"]:.3f}')
    ax2.axvline(x=weight_results['spearman_mean'], color='b', linestyle='--', linewidth=2,
                label=f'平均值: {weight_results["spearman_mean"]:.3f}')
    ax2.set_xlabel('Spearman相关系数（名次向量）')
    ax2.set_ylabel('频次')
    ax2.set_title('排序一致性分布')
    ax2.legend()

    if param_results and ax3 is not None:
        categories = ['权重变化', '参数变化']
        stability_values = [weight_results['topk_overlap_mean'], param_results.get('topk_overlap_mean', 0.0)]
        bars = ax3.bar(categories, stability_values, color=['skyblue', 'lightgreen'])
        ax3.axhline(y=75, color='r', linestyle='--', alpha=0.5, label='参考阈值 (75%)')
        ax3.set_ylabel(f'Top-{topk}重合率 (%)')
        ax3.set_title('稳定性对比：权重扰动 vs 参数扰动')
        ax3.set_ylim([0, 100])
        ax3.legend()
        for bar, value in zip(bars, stability_values):
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., h + 1, f'{value:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# =============================================================================
# 18. 保存结果
# =============================================================================
def save_topk_pareto_to_csv(final_pareto_front, out_dir, top_k=10):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"final_pareto_top{top_k}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "CT", "LoadSTD", "QLoss",
                         "StationLoad_0", "StationLoad_1", "StationLoad_2",
                         "StationLoad_3", "StationLoad_4", "StationLoad_5"])
        if not final_pareto_front:
            return path
        sorted_front = sorted(final_pareto_front, key=lambda x: x[1][0])[:top_k]
        for i, (sol, obj) in enumerate(sorted_front, start=1):
            ct, load_std, qloss = obj
            st = compute_station_times(sol)
            writer.writerow([i, ct, load_std, qloss,
                             float(st[0]), float(st[1]), float(st[2]),
                             float(st[3]), float(st[4]), float(st[5])])
    return path


def save_environment_info(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "hardware_software_env.txt")
    info = get_environment_info()
    with open(path, "w", encoding="utf-8") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    return path


# 19. 主程序

def save_full_pareto_to_csv(final_pareto_front, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "final_pareto_front_full.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "CT", "LoadSTD", "QLoss",
                         "station_loads", "station_assignment", "tool_assignment"])
        for i, (sol, obj) in enumerate(sorted(final_pareto_front, key=lambda x: (x[1][0], x[1][1], x[1][2])), start=1):
            sa, _, ta = sol
            loads = compute_station_times(sol)
            writer.writerow([
                i,
                float(obj[0]), float(obj[1]), float(obj[2]),
                "[" + ",".join([f"{x:.6f}" for x in loads.tolist()]) + "]",
                "[" + ",".join(map(str, sa.tolist())) + "]",
                "[" + ",".join(map(str, ta.tolist())) + "]",
            ])
    return path

def save_history_to_csv(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "optimization_history.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "best_CT", "best_LoadSTD", "best_QLoss", "archive_size"])
        for i in range(len(history["best_ct"])):
            writer.writerow([
                i + 1,
                history["best_ct"][i],
                history["best_load_std"][i],
                history["best_qloss"][i],
                history["archive_size"][i],
            ])
    return path

def save_summary_json(data, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def save_weight_summary_csv(weight_results, out_dir):
    path = os.path.join(out_dir, "weight_sensitivity_summary.csv")
    if not weight_results:
        return path
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key in [
            "n_solutions", "n_samples", "top_k",
            "topk_overlap_mean", "topk_overlap_median",
            "spearman_mean", "spearman_median"
        ]:
            if key in weight_results:
                w.writerow([key, weight_results[key]])
        if "topk_overlap_p10_p90" in weight_results:
            w.writerow(["topk_overlap_p10", weight_results["topk_overlap_p10_p90"][0]])
            w.writerow(["topk_overlap_p90", weight_results["topk_overlap_p10_p90"][1]])
        if "spearman_p10_p90" in weight_results:
            w.writerow(["spearman_p10", weight_results["spearman_p10_p90"][0]])
            w.writerow(["spearman_p90", weight_results["spearman_p10_p90"][1]])
    return path


def save_parameter_summary_csv(param_results, out_dir):
    path = os.path.join(out_dir, "parameter_sensitivity_summary.csv")
    if not param_results:
        return path
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key in [
            "n_solutions", "n_scenarios", "top_k",
            "topk_overlap_mean", "topk_overlap_std",
            "spearman_mean", "spearman_std"
        ]:
            if key in param_results:
                w.writerow([key, param_results[key]])
    return path


def save_plot_from_callable(plot_func, save_path, *args, **kwargs):
    plt.close('all')
    plot_func(*args, **kwargs)
    fig = plt.gcf()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GitHub-ready reproducibility script for IQEA with normalized QLoss and weight/parameter sensitivity analysis."
    )
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--n-obs-base", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--out-dir", type=str, default="results/iqea_normalized_qloss_sensitivity")
    parser.add_argument("--topk-save", type=int, default=10)
    parser.add_argument("--weight-samples", type=int, default=2000)
    parser.add_argument("--sensitivity-top-k", type=int, default=5)
    parser.add_argument("--n-scenarios", type=int, default=27)
    parser.add_argument("--weight-random-state", type=int, default=2025)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    env_path = save_environment_info(str(out_dir))

    epsilon_archive, final_pareto_front, history, cost_stats = quantum_evolutionary_optimization(
        pop_size=args.pop_size,
        max_iter=args.max_iter,
        n_obs_base=args.n_obs_base,
        verbose=args.verbose
    )

    final_sol, final_obj, maj_stats = select_final_solution_majority(final_pareto_front)
    quality_diag = diagnose_quality_components(final_pareto_front)
    weight_results = weight_sensitivity_analysis(
        final_pareto_front,
        base_weights=BASE_WEIGHTS.tolist(),
        n_samples=args.weight_samples,
        top_k=args.sensitivity_top_k,
        random_state=args.weight_random_state
    )
    param_results = parameter_sensitivity_analysis(
        final_pareto_front,
        n_scenarios=args.n_scenarios,
        top_k=args.sensitivity_top_k
    )

    topk_csv = save_topk_pareto_to_csv(final_pareto_front, str(out_dir), top_k=args.topk_save)
    full_front_csv = save_full_pareto_to_csv(final_pareto_front, str(out_dir))
    history_csv = save_history_to_csv(history, str(out_dir))
    weight_csv = save_weight_summary_csv(weight_results, str(out_dir))
    param_csv = save_parameter_summary_csv(param_results, str(out_dir))

    summary_data = {
        "run_seed": int(args.seed),
        "global_random_seed": int(GLOBAL_RANDOM_SEED),
        "algorithm_profile": "safe_speedup_preserve_distribution_consistent",
        "target_expansion": float(TARGET_EXPANSION),
        "base_weights": BASE_WEIGHTS.tolist(),
        "base_quality_params": {k: float(v) for k, v in BASE_QUALITY_PARAMS.items()},
        "cost_stats": cost_stats,
        "epsilon_archive_size": int(len(epsilon_archive)),
        "pareto_solutions_count": int(len(final_pareto_front)),
        "majority_selected_solution": {
            "CT": float(final_obj[0]) if final_obj is not None else None,
            "LoadSTD": float(final_obj[1]) if final_obj is not None else None,
            "QLoss": float(final_obj[2]) if final_obj is not None else None,
            "stats": maj_stats,
        },
        "quality_diagnostics": quality_diag,
        "weight_sensitivity": weight_results,
        "parameter_sensitivity": param_results,
    }
    summary_json = save_summary_json(summary_data, str(out_dir), "run_summary.json")

    if not args.no_plots:
        save_plot_from_callable(plot_optimization_history, out_dir / "optimization_history.png", history)
        if final_pareto_front:
            save_plot_from_callable(plot_3d_pareto_front, out_dir / "pareto_front_3d.png", final_pareto_front)
            save_plot_from_callable(plot_2d_pareto_projections, out_dir / "pareto_front_2d.png", final_pareto_front)
        if weight_results:
            save_plot_from_callable(plot_sensitivity_results, out_dir / "sensitivity_results.png", weight_results, param_results)

    print("=" * 90)
    print("IQEA normalized-QLoss sensitivity experiment completed.")
    print(f"Output directory          : {out_dir.resolve()}")
    print(f"Top-k front CSV           : {topk_csv}")
    print(f"Full front CSV            : {full_front_csv}")
    print(f"Optimization history CSV  : {history_csv}")
    print(f"Weight sensitivity CSV    : {weight_csv}")
    print(f"Parameter sensitivity CSV : {param_csv}")
    print(f"Summary JSON              : {summary_json}")
    print(f"Environment info          : {env_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
