import argparse
import csv
import os
import platform
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

"""Three-objective IQEA for welding-line comparison experiments."""

# =========================
# Problem definition
# =========================
NUM_TASKS = 24
NUM_STATIONS = 6

TASK_TIMES = [
    55, 65, 55, 45, 45, 35, 55, 160, 35, 70,
    100, 80, 35, 60, 60, 160, 35, 70, 450, 40,
    35, 420, 40, 35,
]

ALLOWED_TOOLS: List[set] = []
for task_idx in range(NUM_TASKS):
    task_id = task_idx + 1
    if task_id in {1, 4, 5, 8, 9, 13, 17, 24}:
        ALLOWED_TOOLS.append({0})
    elif task_id in {2, 3, 6, 10}:
        ALLOWED_TOOLS.append({0, 1})
    else:
        ALLOWED_TOOLS.append({2})

TOOL_TYPES = [random.choice(list(tools)) for tools in ALLOWED_TOOLS]
NUM_TOOL_TYPES = len(set(TOOL_TYPES))

TOOL_SWITCH_COST_MATRIX = np.array([
    [0.0, 1.5, 1.8],
    [1.5, 0.0, 1.2],
    [1.8, 1.2, 0.0],
], dtype=float)

PRECEDENCE_CONSTRAINTS = [
    (0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (9, 11), (10, 11), (11, 12), (13, 15), (14, 15), (15, 16),
    (17, 18), (18, 19), (8, 20), (12, 20), (16, 20), (19, 20),
    (20, 21), (21, 22), (22, 23),
]

EPS = 1e-12

# Internal epsilon-dominance parameters
GLOBAL_EPS_CT = 0.5
GLOBAL_EPS_STD = 0.25
LOCAL_EPS_CT = 5.0
LOCAL_EPS_STD = 3.0

MAX_ARCHIVE_SIZE = 20
ELITE_TOP_K = 10
IDEAL_POINT_WEIGHTS = (0.4, 0.2, 0.4)

# Quality-loss parameters (average-value version)
QUAL_ALPHA = 0.2
QUAL_BETA = 0.5
QUAL_GAMMA = 0.3
TARGET_EXPANSION = 0.05
POSITIVE_WEIGHT = 1.5
NEGATIVE_WEIGHT = 0.8

# Cached quality surrogates
TASK_TIMES_ARRAY = np.asarray(TASK_TIMES, dtype=float)
MAX_TASK_TIME = float(np.max(TASK_TIMES_ARRAY))
TASK_SCALE_CACHE = 1.0 + (TASK_TIMES_ARRAY / (MAX_TASK_TIME + EPS)) * 0.2
STATION_ADJUST_CACHE = np.asarray([1.0 + ((s % 3) - 1) * 0.2 for s in range(NUM_STATIONS)], dtype=float)
_TOOL_BASE_R = np.asarray([0.8, 0.6, 0.5], dtype=float)
_TOOL_BASE_D = np.asarray([0.06, 0.08, 0.10], dtype=float)
_TOOL_BASE_E = np.asarray([0.02, -0.01, 0.03], dtype=float)
QUALITY_R_CACHE = np.zeros((NUM_TASKS, NUM_STATIONS, 3), dtype=float)
QUALITY_D_CACHE = np.zeros((NUM_TASKS, NUM_STATIONS, 3), dtype=float)
QUALITY_E_CACHE = np.zeros((NUM_TASKS, NUM_STATIONS, 3), dtype=float)
for _task in range(NUM_TASKS):
    for _station in range(NUM_STATIONS):
        _scale = TASK_SCALE_CACHE[_task] * STATION_ADJUST_CACHE[_station]
        QUALITY_R_CACHE[_task, _station, :] = _TOOL_BASE_R * _scale
        QUALITY_D_CACHE[_task, _station, :] = np.clip(_TOOL_BASE_D * _scale, 0.0, 0.5)
        QUALITY_E_CACHE[_task, _station, :] = _TOOL_BASE_E * _scale


# =========================
# Runtime statistics
# =========================
COST_STATS: Dict[str, float] = {}


def reset_cost_stats() -> None:
    global COST_STATS
    COST_STATS = {
        "n_obj_eval_total": 0,
        "n_local_search_calls": 0,
        "time_local_search_sec": 0.0,
        "runtime_sec": 0.0,
    }


def get_cost_stats_snapshot() -> Dict[str, float]:
    stats = dict(COST_STATS)
    if stats["runtime_sec"] > 0:
        stats["local_search_time_ratio_pct"] = (
            stats["time_local_search_sec"] / stats["runtime_sec"] * 100.0
        )
    else:
        stats["local_search_time_ratio_pct"] = 0.0
    return stats


def get_environment_info() -> Dict[str, str]:
    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
    }


# =========================
# Core objective functions
# =========================
def calculate_quality_loss(solution) -> float:
    station_assignment, _, tool_assignment = solution
    if NUM_TASKS == 0:
        return 0.0

    stations = np.asarray(station_assignment, dtype=int)
    tools = np.asarray(tool_assignment, dtype=int)
    task_idx = np.arange(NUM_TASKS, dtype=int)

    total_roughness = float(np.sum(QUALITY_R_CACHE[task_idx, stations, tools]))
    total_defect_rate = float(np.sum(QUALITY_D_CACHE[task_idx, stations, tools]))
    total_expansion = float(np.sum(QUALITY_E_CACHE[task_idx, stations, tools]))

    avg_roughness = total_roughness / NUM_TASKS
    avg_defect_rate = total_defect_rate / NUM_TASKS
    avg_expansion = total_expansion / NUM_TASKS

    roughness_loss = avg_roughness
    defect_loss = avg_defect_rate
    expansion_diff = avg_expansion - TARGET_EXPANSION
    expansion_loss = (
        expansion_diff * POSITIVE_WEIGHT
        if expansion_diff > 0
        else abs(expansion_diff) * NEGATIVE_WEIGHT
    )
    return QUAL_ALPHA * roughness_loss + QUAL_BETA * defect_loss + QUAL_GAMMA * expansion_loss


def evaluate_welding_objectives_with_penalty(solution):
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


# =========================
# Pareto utilities
# =========================
def dominates_eps(a, b) -> bool:
    ct_dominated = a[0] <= b[0] + GLOBAL_EPS_CT
    std_dominated = a[1] <= b[1] + GLOBAL_EPS_STD
    qloss_dominated = a[2] < b[2]
    all_dominated = ct_dominated and std_dominated and qloss_dominated
    ct_strict = a[0] < b[0] - GLOBAL_EPS_CT
    std_strict = a[1] < b[1] - GLOBAL_EPS_STD
    qloss_strict = a[2] < b[2]
    any_strict = ct_strict or std_strict or qloss_strict
    return all_dominated and any_strict


def dominates_standard(a, b) -> bool:
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


def majority_beats(obj_a, obj_b, k=2) -> bool:
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


# =========================
# IQEA search operators
# =========================
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


def quantum_evolutionary_optimization(pop_size=30, max_iter=100, n_obs_base=3, verbose=False):
    reset_cost_stats()
    t_run_start = time.perf_counter()

    num_qubits = NUM_TASKS * (NUM_STATIONS + NUM_TOOL_TYPES + 1)
    Q = np.zeros((pop_size, num_qubits, 2))
    Q[:, :, 0] = Q[:, :, 1] = 1 / np.sqrt(2)

    epsilon_archive = []
    history = {"best_ct": [], "best_load_std": [], "best_qloss": [], "archive_size": []}

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
            history["best_ct"].append(archive_objs[:, 0].min())
            history["best_load_std"].append(archive_objs[:, 1].min())
            history["best_qloss"].append(archive_objs[:, 2].min())
        else:
            history["best_ct"].append(np.inf)
            history["best_load_std"].append(np.inf)
            history["best_qloss"].append(np.inf)
        history["archive_size"].append(len(epsilon_archive))

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


# =========================
# Comparison metrics: HV / IGD
# =========================
def front_to_array(front) -> np.ndarray:
    if not front:
        return np.empty((0, 3), dtype=float)
    return np.asarray([obj for _, obj in front], dtype=float)


def load_reference_front(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None or not os.path.exists(path):
        return None
    data = np.genfromtxt(path, delimiter=",", names=True)
    required = {"CT", "LoadSTD", "QLoss"}
    if not required.issubset(set(data.dtype.names or [])):
        raise ValueError("reference_front.csv must contain columns: CT, LoadSTD, QLoss")
    return np.column_stack([data["CT"], data["LoadSTD"], data["QLoss"]]).astype(float)


def filter_nondominated_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3)
    keep = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not keep[i]:
            continue
        for j in range(len(points)):
            if i == j or not keep[j]:
                continue
            if np.all(points[j] <= points[i] + EPS) and np.any(points[j] < points[i] - EPS):
                keep[i] = False
                break
    uniq = np.unique(points[keep], axis=0)
    return uniq


def get_reference_point(reference_front: np.ndarray, approx_front: np.ndarray) -> np.ndarray:
    combined = reference_front if approx_front.size == 0 else np.vstack([reference_front, approx_front])
    mins = np.min(combined, axis=0)
    maxs = np.max(combined, axis=0)
    ranges = np.maximum(maxs - mins, 1.0)
    return maxs + 0.1 * ranges


def compute_hypervolume_3d(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact 3D hypervolume for minimization via grid-union decomposition."""
    if points.size == 0:
        return 0.0
    points = filter_nondominated_points(points)
    points = points[np.all(points < ref_point - EPS, axis=1)]
    if points.size == 0:
        return 0.0

    xs = np.sort(np.unique(np.concatenate([points[:, 0], [ref_point[0]]])))
    ys = np.sort(np.unique(np.concatenate([points[:, 1], [ref_point[1]]])))
    zs = np.sort(np.unique(np.concatenate([points[:, 2], [ref_point[2]]])))

    hv = 0.0
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        xm = 0.5 * (x0 + x1)
        for j in range(len(ys) - 1):
            y0, y1 = ys[j], ys[j + 1]
            ym = 0.5 * (y0 + y1)
            for k in range(len(zs) - 1):
                z0, z1 = zs[k], zs[k + 1]
                zm = 0.5 * (z0 + z1)
                dominated = np.any(np.all(points <= np.array([xm, ym, zm]) + EPS, axis=1))
                if dominated:
                    hv += (x1 - x0) * (y1 - y0) * (z1 - z0)
    return float(hv)


def compute_igd(approx_front: np.ndarray, reference_front: np.ndarray) -> float:
    if approx_front.size == 0 or reference_front is None or reference_front.size == 0:
        return float("nan")
    combined = np.vstack([reference_front, approx_front])
    mins = np.min(combined, axis=0)
    maxs = np.max(combined, axis=0)
    ranges = np.maximum(maxs - mins, EPS)
    ref_norm = (reference_front - mins) / ranges
    approx_norm = (approx_front - mins) / ranges
    dists = []
    for rp in ref_norm:
        dist = np.min(np.linalg.norm(approx_norm - rp, axis=1))
        dists.append(dist)
    return float(np.mean(dists))


# =========================
# File outputs
# =========================
def save_environment_info(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "hardware_software_env.txt")
    info = get_environment_info()
    with open(path, "w", encoding="utf-8") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    return path


def save_seeds(seeds: Sequence[int], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "seeds.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed"])
        for i, seed in enumerate(seeds, start=1):
            w.writerow([i, seed])
    return path


def save_front_csv(final_pareto_front, run_id: int, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"run_{run_id:02d}_final_pareto_front.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["CT", "LoadSTD", "QLoss"])
        if final_pareto_front:
            front = sorted([obj for _, obj in final_pareto_front], key=lambda x: (x[0], x[1], x[2]))
            for obj in front:
                w.writerow([float(obj[0]), float(obj[1]), float(obj[2])])
    return path


def save_per_run_metrics(rows: List[List], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "per_run_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "run", "seed", "CT", "LoadSTD", "QLoss",
            "HV", "IGD", "runtime_sec", "n_obj_eval_total"
        ])
        w.writerows(rows)
    return path


def save_summary_statistics(rows: List[Tuple[str, float, float]], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary_statistics.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "std"])
        for row in rows:
            w.writerow(row)
    return path


def build_summary_rows(per_run_rows: List[List]) -> List[Tuple[str, float, float]]:
    arr = np.asarray(per_run_rows, dtype=object)
    summary = []
    metric_columns = {
        "CT": 2,
        "LoadSTD": 3,
        "QLoss": 4,
        "HV": 5,
        "IGD": 6,
        "runtime_sec": 7,
        "n_obj_eval_total": 8,
    }
    for name, idx in metric_columns.items():
        vals = np.asarray(arr[:, idx], dtype=float)
        summary.append((name, float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0))
    return summary


# =========================
# Main entry
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IQEA comparison script for review reproducibility.")
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--n-obs-base", type=int, default=3)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--out-dir", type=str, default="results/iqea")
    parser.add_argument(
        "--reference-front",
        type=str,
        default=None,
        help="CSV file with columns CT,LoadSTD,QLoss. Required for HV/IGD.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    env_path = save_environment_info(args.out_dir)
    seeds = [args.base_seed + run for run in range(1, args.num_runs + 1)]
    seeds_path = save_seeds(seeds, args.out_dir)

    reference_front = load_reference_front(args.reference_front)
    if reference_front is not None:
        reference_front = filter_nondominated_points(reference_front)

    per_run_rows: List[List] = []

    for run, seed in enumerate(seeds, start=1):
        random.seed(seed)
        np.random.seed(seed)

        print("\n" + "-" * 90)
        print(f"IQEA run {run:02d}/{args.num_runs:02d} starts...")

        epsilon_archive, final_pareto_front, _history, cost_stats = quantum_evolutionary_optimization(
            pop_size=args.pop_size,
            max_iter=args.max_iter,
            n_obs_base=args.n_obs_base,
            verbose=args.verbose,
        )

        front_csv = save_front_csv(final_pareto_front, run, args.out_dir)

        if final_pareto_front:
            _final_sol, final_obj, _maj_stats = select_final_solution_majority(final_pareto_front)
            ct_f, std_f, ql_f = final_obj
            approx_front = front_to_array(final_pareto_front)
            if reference_front is not None:
                ref_point = get_reference_point(reference_front, approx_front)
                hv = compute_hypervolume_3d(approx_front, ref_point)
                igd = compute_igd(approx_front, reference_front)
            else:
                hv = float("nan")
                igd = float("nan")

            print(
                f"CT={ct_f:.2f}, LoadSTD={std_f:.2f}, QLoss={ql_f:.4f}, "
                f"HV={hv:.6f}, IGD={igd:.6f}, Runtime={cost_stats['runtime_sec']:.4f}s, "
                f"ObjEval={cost_stats['n_obj_eval_total']}, |front|={len(final_pareto_front)}"
            )
        else:
            ct_f, std_f, ql_f = np.nan, np.nan, np.nan
            hv, igd = np.nan, np.nan
            print(
                f"Run {run:02d}: final Pareto front empty | Runtime={cost_stats['runtime_sec']:.4f}s | "
                f"ObjEval={cost_stats['n_obj_eval_total']}"
            )

        per_run_rows.append([
            run,
            seed,
            float(ct_f),
            float(std_f),
            float(ql_f),
            float(hv),
            float(igd),
            float(cost_stats["runtime_sec"]),
            float(cost_stats["n_obj_eval_total"]),
        ])

        print(f"Saved front: {front_csv}")

    per_run_csv = save_per_run_metrics(per_run_rows, args.out_dir)
    summary_csv = save_summary_statistics(build_summary_rows(per_run_rows), args.out_dir)

    print("\n" + "=" * 90)
    print("IQEA comparison experiment completed.")
    print(f"Per-run metrics saved: {per_run_csv}")
    print(f"Summary statistics saved: {summary_csv}")
    print(f"Environment info saved: {env_path}")
    print(f"Seeds saved: {seeds_path}")
    if reference_front is None:
        print("HV and IGD were not computed because --reference-front was not provided.")
    else:
        print(f"Reference front used: {args.reference_front}")
    print("=" * 90)


if __name__ == "__main__":
    main()
