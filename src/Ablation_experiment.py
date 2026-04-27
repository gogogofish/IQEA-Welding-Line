import numpy as np
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import hashlib


# =========================
# 0) Problem Setup
# =========================
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

# Initial tool types only used when extended encoding is enabled
TOOL_TYPES_INIT = [random.choice(list(tools)) for tools in ALLOWED_TOOLS]
NUM_TOOL_TYPES = len(set(TOOL_TYPES_INIT))

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

# epsilon-dominance parameters
EPS_CT = 0.5
EPS_STD = 0.25

MAX_ARCHIVE_SIZE = 20

# Quality parameters
QUAL_ALPHA = 0.2
QUAL_BETA = 0.5
QUAL_GAMMA = 0.3
TARGET_EXPANSION = 0.025
POSITIVE_WEIGHT = 1.5
NEGATIVE_WEIGHT = 0.8


# =========================
# 1) Utilities: Quality, Objectives, Constraints
# =========================
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

    avg_roughness = total_roughness / NUM_TASKS
    avg_defect_rate = total_defect_rate / NUM_TASKS
    avg_expansion = total_expansion / NUM_TASKS

    roughness_loss = avg_roughness
    defect_loss = avg_defect_rate
    expansion_diff = avg_expansion - TARGET_EXPANSION
    expansion_loss = expansion_diff * POSITIVE_WEIGHT if expansion_diff > 0 else abs(expansion_diff) * NEGATIVE_WEIGHT

    return QUAL_ALPHA * roughness_loss + QUAL_BETA * defect_loss + QUAL_GAMMA * expansion_loss


def evaluate_welding_objectives_with_penalty(solution):
    station_assignment, station_sequences, tool_assignment = solution
    penalty = 0.0

    for pre, post in PRECEDENCE_CONSTRAINTS:
        if station_assignment[pre] > station_assignment[post]:
            penalty += 15.0
        elif station_assignment[pre] == station_assignment[post]:
            seq = station_sequences[int(station_assignment[pre])]
            try:
                if seq.index(pre) > seq.index(post):
                    penalty += 10.0
            except ValueError:
                penalty += 5.0

    station_times = np.zeros(NUM_STATIONS)
    for s in range(NUM_STATIONS):
        seq = station_sequences[s]
        if not seq:
            continue
        station_times[s] += TASK_TIMES[seq[0]]
        current_tool = int(tool_assignment[seq[0]])
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


def has_empty_station(solution) -> bool:
    _, station_sequences, _ = solution
    return any(len(seq) == 0 for seq in station_sequences)


def dominates_eps(a, b):
    # a epsilon-dominates b (CT/STD tolerant, QLoss strict)
    ct_dom = a[0] <= b[0] + EPS_CT
    std_dom = a[1] <= b[1] + EPS_STD
    ql_dom = a[2] < b[2]
    all_dom = ct_dom and std_dom and ql_dom

    ct_strict = a[0] < b[0] - EPS_CT
    std_strict = a[1] < b[1] - EPS_STD
    ql_strict = a[2] < b[2]
    any_strict = ct_strict or std_strict or ql_strict
    return all_dom and any_strict


def calculate_crowding_distance(archive: List[Tuple[Any, Tuple[float, float, float]]]) -> List[float]:
    n = len(archive)
    if n == 0:
        return []
    objs = np.array([obj for _, obj in archive], dtype=float)
    cd = np.zeros(n, dtype=float)

    for m in range(3):
        idx = np.argsort(objs[:, m])
        cd[idx[0]] = np.inf
        cd[idx[-1]] = np.inf
        if n > 2:
            minv = objs[idx[0], m]
            maxv = objs[idx[-1], m]
            rng = maxv - minv
            if rng > EPS:
                for k in range(1, n - 1):
                    cd[idx[k]] += (objs[idx[k + 1], m] - objs[idx[k - 1], m]) / rng
    return cd.tolist()


def update_archive_pareto(archive, cand_solution, cand_obj):
    for _, obj in archive:
        if dominates_eps(obj, cand_obj):
            return archive
    new_archive = [(sol, obj) for sol, obj in archive if not dominates_eps(cand_obj, obj)]
    new_archive.append((cand_solution, cand_obj))

    if len(new_archive) > MAX_ARCHIVE_SIZE:
        cd = calculate_crowding_distance(new_archive)
        keep = np.argsort(cd)[::-1][:MAX_ARCHIVE_SIZE]
        new_archive = [new_archive[i] for i in keep]
    return new_archive


# =========================
# 2) Config for Ablations
# =========================
@dataclass
class IQEAConfig:
    name: str
    pop_size: int = 30
    max_iter: int = 100
    n_obs_base: int = 3

    use_extended_encoding: bool = True  # False for Abl-1
    n_obs_fixed_one: bool = False       # True for Abl-2
    objective_mode: str = "multi"       # "multi" or "weighted_sum" for Abl-3a
    update_mode: str = "improved"       # "improved" or "classic" for Abl-4

    # weighted-sum weights (Abl-3a)
    ws_w: Tuple[float, float, float] = (2/5, 2/5, 1/5)

    # Monte Carlo HV samples
    hv_mc_samples: int = 20000


def count_qubits(cfg: IQEAConfig) -> int:
    # Station assignment: NUM_TASKS * NUM_STATIONS
    base = NUM_TASKS * NUM_STATIONS
    if cfg.use_extended_encoding:
        # tool assignment: NUM_TASKS * NUM_TOOL_TYPES
        # order qubit per task: NUM_TASKS
        return base + NUM_TASKS * NUM_TOOL_TYPES + NUM_TASKS
    return base


def stable_name_hash(name: str) -> int:
    # deterministic across runs/processes (avoid Python's randomized hash())
    h = hashlib.md5(name.encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


def is_exact_full_cfg(cfg: IQEAConfig) -> bool:
    return (
        cfg.pop_size == 30 and
        cfg.max_iter == 100 and
        cfg.n_obs_base == 3 and
        cfg.use_extended_encoding is True and
        cfg.n_obs_fixed_one is False and
        cfg.objective_mode == "multi" and
        cfg.update_mode == "improved"
    )


def strict_dominates_exact(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.all(a <= b + EPS) and np.any(a < b - EPS)


def strict_pareto_filter_exact(solution_obj_list):
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
            if strict_dominates_exact(obj_j, obj_i):
                dominated = True
                break
            if np.allclose(np.asarray(obj_i, dtype=float), np.asarray(obj_j, dtype=float), atol=1e-12, rtol=0.0):
                if j < i:
                    duplicate = True
                    break
        if not dominated and not duplicate:
            filtered.append((sol_i, obj_i))
    return filtered


def select_representative_solutions_exact(front, w=(0.6, 0.2, 0.2)):
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


def select_topk_elite_archive_solutions_exact(epsilon_archive, top_k=10, w=(0.4, 0.2, 0.4)):
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


def rebuild_epsilon_archive_from_candidates_exact(candidates):
    archive = []
    for sol, obj in candidates:
        archive = update_archive_pareto(archive, sol, obj)
    return archive


def observe_individual_exact(Q_j, n_obs):
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


def update_Q_exact(Q, guided_archive, X_obs, t, max_iter):
    pop_size, num_qubits, _ = Q.shape
    new_Q = np.copy(Q)
    valid_size = len(guided_archive)
    if valid_size == 0:
        return new_Q
    progress = t / max_iter

    reps = select_representative_solutions_exact(guided_archive)
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
    for idx, j in enumerate(update_indices):
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


def local_qloss_improvement_exact(solution, obj_values):
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
            ct_ok = test_ct <= ct + 5.0
            std_ok = test_std <= load_std + 3.0
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


def run_iqea_full_exact(seed: int):
    np.random.seed(seed)
    random.seed(seed)

    pop_size = 30
    max_iter = 100
    n_obs_base = 3
    num_qubits = NUM_TASKS * (NUM_STATIONS + NUM_TOOL_TYPES + 1)
    Q = np.zeros((pop_size, num_qubits, 2))
    Q[:, :, 0] = Q[:, :, 1] = 1 / np.sqrt(2)

    epsilon_archive = []

    for t in range(max_iter):
        solutions = []
        original_objs = []
        penalized_objs = []
        X_obs = []
        progress = t / max_iter
        n_obs = n_obs_base + int(2 * progress) if progress < 1.0 else n_obs_base + 2

        for j in range(pop_size):
            sol, obs_bits, orig_obj, penalized_obj = observe_individual_exact(Q[j], n_obs)
            if not has_empty_station(sol):
                solutions.append(sol)
                original_objs.append(orig_obj)
                penalized_objs.append(penalized_obj)
                X_obs.append(obs_bits)

        X_obs = np.array(X_obs) if X_obs else np.empty((0, num_qubits))

        for sol, orig_obj in zip(solutions, original_objs):
            if not np.any(np.isinf(orig_obj)):
                epsilon_archive = update_archive_pareto(epsilon_archive, sol, orig_obj)

        if progress > 0.5 and epsilon_archive:
            elite_solutions = select_topk_elite_archive_solutions_exact(
                epsilon_archive, top_k=10, w=(0.4, 0.2, 0.4)
            )
            elite_ids = {id(sol) for sol, _ in elite_solutions}
            merged_candidates = []
            for sol, obj in epsilon_archive:
                if id(sol) in elite_ids:
                    improved_sol, improved_obj = local_qloss_improvement_exact(sol, obj)
                    merged_candidates.append((improved_sol, improved_obj))
                else:
                    merged_candidates.append((sol, obj))
            epsilon_archive = rebuild_epsilon_archive_from_candidates_exact(merged_candidates)

        guided_archive = [(sol, pen_obj) for sol, pen_obj in zip(solutions, penalized_objs)]
        Q = update_Q_exact(Q, guided_archive, X_obs, t, max_iter)

    final_pareto_front = strict_pareto_filter_exact(epsilon_archive)
    return np.array([obj for _, obj in final_pareto_front], dtype=float) if final_pareto_front else np.empty((0, 3))


# =========================
# 3) Observation
# =========================
def build_solution_from_observation(Q_j, cfg: IQEAConfig):
    """
    One observation sample -> (solution, observed_bits, original_obj, penalized_obj)
    """
    ptr = 0
    observed_bits = []
    station_assignment = np.zeros(NUM_TASKS, dtype=int)

    # 3.1 station assignment probs from qubits
    for task in range(NUM_TASKS):
        probs = []
        for s in range(NUM_STATIONS):
            alpha, beta = Q_j[ptr]
            p0 = float(np.abs(alpha) ** 2)
            probs.append(p0)
            observed_bits.append(0 if np.random.rand() < p0 else 1)
            ptr += 1
        probs = np.array(probs, dtype=float)
        probs = probs / probs.sum() if probs.sum() > EPS else (np.ones(NUM_STATIONS) / NUM_STATIONS)
        station_assignment[task] = np.random.choice(NUM_STATIONS, p=probs)

    # 3.2 tool assignment
    if cfg.use_extended_encoding:
        tool_assignment = np.zeros(NUM_TASKS, dtype=int)
        for task in range(NUM_TASKS):
            probs = []
            for tt in range(NUM_TOOL_TYPES):
                alpha, beta = Q_j[ptr]
                p0 = float(np.abs(alpha) ** 2)
                probs.append(p0)
                observed_bits.append(0 if np.random.rand() < p0 else 1)
                ptr += 1
            probs = np.array(probs, dtype=float)
            if probs.sum() < EPS:
                chosen = np.random.choice(list(ALLOWED_TOOLS[task]))
            else:
                probs /= probs.sum()
                chosen = int(np.random.choice(NUM_TOOL_TYPES, p=probs))
                if chosen not in ALLOWED_TOOLS[task]:
                    allowed = sorted(ALLOWED_TOOLS[task])
                    chosen = min(allowed, key=lambda x: abs(x - chosen))
            tool_assignment[task] = chosen
    else:
        # Abl-1: fixed tool = min allowed
        tool_assignment = np.array([min(ALLOWED_TOOLS[i]) for i in range(NUM_TASKS)], dtype=int)

    # 3.3 station sequences (order)
    station_sequences = [[] for _ in range(NUM_STATIONS)]
    if cfg.use_extended_encoding:
        # order qubits exist -> use random keys but still advance ptr & record bits
        rand_keys = np.zeros(NUM_TASKS, dtype=float)
        for task in range(NUM_TASKS):
            alpha, beta = Q_j[ptr]
            observed_bits.append(0 if np.random.rand() < abs(alpha) ** 2 else 1)
            rand_keys[task] = np.random.rand()
            ptr += 1

        for task, s in enumerate(station_assignment):
            station_sequences[int(s)].append((task, rand_keys[task]))
        for s in range(NUM_STATIONS):
            station_sequences[s].sort(key=lambda x: x[1])
            station_sequences[s] = [t for t, _ in station_sequences[s]]
    else:
        # Abl-1: fixed order within station by task id
        for task, s in enumerate(station_assignment):
            station_sequences[int(s)].append(task)
        for s in range(NUM_STATIONS):
            station_sequences[s].sort()

    sol = (station_assignment, station_sequences, tool_assignment)
    orig_obj, pen_obj, _ = evaluate_welding_objectives_with_penalty(sol)
    return sol, np.array(observed_bits, dtype=int), orig_obj, pen_obj


def observe_individual(Q_j, cfg: IQEAConfig, n_obs: int):
    """
    Multiple independent observations -> choose best
      - multi: choose the candidate dominated least by others (on penalized objs)
      - weighted_sum: choose minimal weighted-sum scalar (on penalized objs, normalized within candidates)
    """
    candidates = []
    for _ in range(n_obs):
        sol, bits, orig_obj, pen_obj = build_solution_from_observation(Q_j, cfg)
        candidates.append((sol, bits, orig_obj, pen_obj))

    if len(candidates) == 1:
        return candidates[0][0], candidates[0][1], candidates[0][2], candidates[0][3]

    if cfg.objective_mode == "multi":
        dominated_counts = []
        for i, (_, _, _, pen_i) in enumerate(candidates):
            cnt = 0
            for j, (_, _, _, pen_j) in enumerate(candidates):
                if i != j and dominates_eps(pen_j, pen_i):
                    cnt += 1
            dominated_counts.append(cnt)
        best = int(np.argmin(dominated_counts))
        return candidates[best][0], candidates[best][1], candidates[best][2], candidates[best][3]

    # weighted_sum selection among candidates (normalize within candidates)
    pen = np.array([c[3] for c in candidates], dtype=float)  # (k,3)
    mins = pen.min(axis=0)
    maxs = pen.max(axis=0)
    rng = np.maximum(maxs - mins, EPS)
    pen_n = (pen - mins) / rng
    w = np.array(cfg.ws_w, dtype=float)
    s = pen_n @ w
    best = int(np.argmin(s))
    return candidates[best][0], candidates[best][1], candidates[best][2], candidates[best][3]


# =========================
# 4) Quantum Update (Improved vs Classic)
# =========================
def solution_to_ref_bits(ref_solution, cfg: IQEAConfig) -> np.ndarray:
    """
    Encode reference solution into the bit layout used by cfg (station only vs full).
    Only required for update_Q.
    """
    station_assignment, station_sequences, tool_assignment = ref_solution
    bits = []

    # station one-hot
    for task in range(NUM_TASKS):
        cs = int(station_assignment[task])
        for s in range(NUM_STATIONS):
            bits.append(1 if s == cs else 0)

    if cfg.use_extended_encoding:
        # tool one-hot
        for task in range(NUM_TASKS):
            ct = int(tool_assignment[task])
            for tt in range(NUM_TOOL_TYPES):
                bits.append(1 if tt == ct else 0)
        # order bit (rough, from rank position)
        for task in range(NUM_TASKS):
            s = int(station_assignment[task])
            seq = station_sequences[s]
            if task in seq and len(seq) > 0:
                rk = seq.index(task)
                bits.append(1 if rk >= len(seq) / 2 else 0)
            else:
                bits.append(0)

    return np.array(bits, dtype=int)


def pick_reference_solution_improved(guided_solutions: List[Any], guided_pen_objs: np.ndarray, cfg: IQEAConfig, progress: float):
    """
    Improved mode reference:
      - multi: use ideal/knee/random from a Pareto-like pool (approx: non-dominated filtering on penalized)
      - weighted_sum: use best scalar in current guided pool
    """
    if len(guided_solutions) == 0:
        # fallback: round-robin stations, min tools, task-id order
        ref_station = np.array([i % NUM_STATIONS for i in range(NUM_TASKS)], dtype=int)
        ref_tool = np.array([min(ALLOWED_TOOLS[i]) for i in range(NUM_TASKS)], dtype=int)
        ref_seq = [[] for _ in range(NUM_STATIONS)]
        for i, s in enumerate(ref_station):
            ref_seq[int(s)].append(i)
        return (ref_station, ref_seq, ref_tool)

    if cfg.objective_mode == "weighted_sum":
        # pick best weighted-sum (normalize within guided)
        mins = guided_pen_objs.min(axis=0)
        maxs = guided_pen_objs.max(axis=0)
        rng = np.maximum(maxs - mins, EPS)
        pen_n = (guided_pen_objs - mins) / rng
        w = np.array(cfg.ws_w, dtype=float)
        s = pen_n @ w
        return guided_solutions[int(np.argmin(s))]

    # multi: approximate a Pareto pool (non-dominated on penalized)
    nd_idx = []
    for i in range(len(guided_pen_objs)):
        dominated = False
        for j in range(len(guided_pen_objs)):
            if i != j and dominates_eps(tuple(guided_pen_objs[j]), tuple(guided_pen_objs[i])):
                dominated = True
                break
        if not dominated:
            nd_idx.append(i)
    pool_idx = nd_idx if len(nd_idx) > 0 else list(range(len(guided_solutions)))
    pool_solutions = [guided_solutions[i] for i in pool_idx]
    pool_objs = guided_pen_objs[pool_idx]

    # ideal-point / knee / random mixing
    p_random = 0.35 * (1 - progress) + 0.05
    p_knee = 0.15 + 0.15 * (1 - progress)
    p_ideal = max(1.0 - p_random - p_knee, 0.0)
    choice = np.random.choice(["ideal", "knee", "random"], p=[p_ideal, p_knee, p_random])

    # normalize objs in pool
    mins = pool_objs.min(axis=0)
    maxs = pool_objs.max(axis=0)
    rng = np.maximum(maxs - mins, EPS)
    norm = (pool_objs - mins) / rng

    if choice == "random":
        return random.choice(pool_solutions)

    # ideal: closest to [0,0,0]
    if choice == "ideal" or len(pool_solutions) == 1:
        d = np.linalg.norm(norm - np.zeros(3), axis=1)
        return pool_solutions[int(np.argmin(d))]

    # knee: max distance to diagonal line
    utopia = np.zeros(3)
    nadir = np.ones(3)
    line_dir = (nadir - utopia) / np.linalg.norm(nadir - utopia)
    projs = (norm @ line_dir.reshape(-1, 1)) * line_dir.reshape(1, -1)
    perp = norm - projs
    dist = np.linalg.norm(perp, axis=1)
    return pool_solutions[int(np.argmax(dist))]


def update_Q_improved(Q, guided_solutions, guided_pen_objs, X_obs, t, max_iter, cfg: IQEAConfig):
    pop_size, num_qubits, _ = Q.shape
    if len(guided_solutions) == 0 or X_obs.shape[0] == 0:
        return Q.copy()

    progress = t / max_iter
    ref_solution = pick_reference_solution_improved(guided_solutions, guided_pen_objs, cfg, progress)
    ref_bits = solution_to_ref_bits(ref_solution, cfg)

    new_Q = Q.copy()
    base_angle = 0.03 * np.pi * np.exp(-t / (max_iter / 2))
    archive_size = len(guided_solutions)

    p_mut_min, p_mut_max = 0.006, 0.06
    size_factor = 1.0 / (1.0 + np.log1p(max(archive_size, 1)))
    p_mut = p_mut_min + (p_mut_max - p_mut_min) * (1 - progress) * size_factor
    phi_max = (0.04 * np.pi) * (1 - progress) + 0.02 * np.pi
    p_reinit, p_flip = 0.4 * p_mut, 0.4 * p_mut

    valid_size = X_obs.shape[0]
    update_indices = np.random.choice(pop_size, size=min(valid_size, pop_size), replace=False)

    for idx, j in enumerate(update_indices):
        xi_bits = X_obs[idx]
        for i in range(num_qubits):
            alpha, beta = Q[j, i]
            xi = int(xi_bits[i])
            ri = int(ref_bits[i])

            if xi == ri:
                direction = 1.0 if np.random.rand() < 0.7 else -1.0
            else:
                direction = -1.0 if np.random.rand() < 0.8 else 1.0

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

            # adaptive mutation
            p1 = float(abs(new_beta) ** 2)
            polarization = abs(p1 - 0.5)
            boost = 1.0 + 1.5 * max(0.0, polarization - 0.35)

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

                norm2 = np.hypot(new_alpha, new_beta)
                if norm2 < EPS:
                    new_alpha = new_beta = 1 / np.sqrt(2)
                else:
                    new_alpha /= norm2
                    new_beta /= norm2

            new_Q[j, i, 0] = new_alpha
            new_Q[j, i, 1] = new_beta

    return new_Q


def update_Q_classic(Q, guided_solutions, guided_pen_objs, X_obs, t, max_iter, cfg: IQEAConfig):
    """
    Classic QEA update (Abl-4):
      - reference: best weighted-sum (normalize within guided)
      - fixed rotation angle
      - no adaptive mutation
    """
    pop_size, num_qubits, _ = Q.shape
    if len(guided_solutions) == 0 or X_obs.shape[0] == 0:
        return Q.copy()

    mins = guided_pen_objs.min(axis=0)
    maxs = guided_pen_objs.max(axis=0)
    rng = np.maximum(maxs - mins, EPS)
    pen_n = (guided_pen_objs - mins) / rng
    w = np.array((1/3, 1/3, 1/3), dtype=float)
    s = pen_n @ w
    ref_solution = guided_solutions[int(np.argmin(s))]
    ref_bits = solution_to_ref_bits(ref_solution, cfg)

    new_Q = Q.copy()
    theta0 = 0.03 * np.pi  # fixed

    valid_size = X_obs.shape[0]
    update_indices = np.random.choice(pop_size, size=min(valid_size, pop_size), replace=False)

    for idx, j in enumerate(update_indices):
        xi_bits = X_obs[idx]
        for i in range(num_qubits):
            alpha, beta = Q[j, i]
            xi = int(xi_bits[i])
            ri = int(ref_bits[i])

            direction = 1.0 if (xi != ri and ri == 1) else (-1.0 if (xi != ri and ri == 0) else 0.0)
            if direction == 0.0:
                continue

            theta = theta0 * direction
            c, s_ = np.cos(theta), np.sin(theta)
            new_alpha = c * alpha - s_ * beta
            new_beta = s_ * alpha + c * beta

            norm = np.hypot(new_alpha, new_beta)
            if norm < EPS:
                continue
            new_Q[j, i, 0] = new_alpha / norm
            new_Q[j, i, 1] = new_beta / norm

    return new_Q


# =========================
# 5) Main IQEA loop 
# =========================
def run_iqea(cfg: IQEAConfig, seed: int):
    if is_exact_full_cfg(cfg):
        return run_iqea_full_exact(seed)

    np.random.seed(seed)
    random.seed(seed)

    num_qubits = count_qubits(cfg)
    Q = np.zeros((cfg.pop_size, num_qubits, 2), dtype=float)
    Q[:, :, 0] = Q[:, :, 1] = 1 / np.sqrt(2)

    # For multi: pareto archive
    pareto_archive = []

    # For weighted_sum: keep top solutions by scalar (final "archive-like" pool)
    top_pool = []

    for t in range(cfg.max_iter):
        progress = t / cfg.max_iter

        if cfg.n_obs_fixed_one:
            n_obs = 1
        else:
            n_obs = cfg.n_obs_base + int(2 * progress) if progress < 1.0 else cfg.n_obs_base + 2

        solutions = []
        orig_objs = []
        pen_objs = []
        X_obs = []

        for j in range(cfg.pop_size):
            sol, bits, orig_obj, pen_obj = observe_individual(Q[j], cfg, n_obs)
            if not has_empty_station(sol):
                solutions.append(sol)
                orig_objs.append(orig_obj)
                pen_objs.append(pen_obj)
                X_obs.append(bits)

        if len(solutions) == 0:
            continue

        orig_objs = np.array(orig_objs, dtype=float)
        pen_objs = np.array(pen_objs, dtype=float)
        X_obs = np.array(X_obs, dtype=int) if len(X_obs) > 0 else np.empty((0, num_qubits), dtype=int)

        if cfg.objective_mode == "multi":
            for sol, obj in zip(solutions, orig_objs):
                pareto_archive = update_archive_pareto(pareto_archive, sol, tuple(obj))
            guided_solutions = solutions
            guided_pen = pen_objs
        else:
            # weighted_sum: maintain top_pool by scalar fitness (normalized per generation)
            mins = orig_objs.min(axis=0)
            maxs = orig_objs.max(axis=0)
            rng = np.maximum(maxs - mins, EPS)
            orig_n = (orig_objs - mins) / rng
            w = np.array(cfg.ws_w, dtype=float)
            scalar = orig_n @ w
            order = np.argsort(scalar)

            keep = order[:min(MAX_ARCHIVE_SIZE, len(order))]
            top_pool = [(solutions[i], tuple(orig_objs[i])) for i in keep]

            guided_solutions = solutions
            guided_pen = pen_objs

        # update quantum population
        if cfg.update_mode == "improved":
            Q = update_Q_improved(Q, guided_solutions, guided_pen, X_obs, t, cfg.max_iter, cfg)
        else:
            Q = update_Q_classic(Q, guided_solutions, guided_pen, X_obs, t, cfg.max_iter, cfg)

    # Return final objective set used for metrics
    if cfg.objective_mode == "multi":
        objs = np.array([obj for _, obj in pareto_archive], dtype=float) if len(pareto_archive) > 0 else np.empty((0, 3))
    else:
        objs = np.array([obj for _, obj in top_pool], dtype=float) if len(top_pool) > 0 else np.empty((0, 3))
    return objs


# =========================
# 6) Metrics: HV (Monte Carlo), objective-space variance, best objectives
# =========================
def normalize_by_bounds(objs: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    rng = np.maximum(maxs - mins, EPS)
    return (objs - mins) / rng


def hv_monte_carlo_3d(points_norm: np.ndarray, n_samples: int, rng_seed: int = 12345) -> float:
    """
    HV in [0,1]^3 for minimization with reference point at (1,1,1).
    Monte Carlo estimate:
      sample u ~ Uniform([0,1]^3), count if exists point p s.t. p <= u elementwise.
    """
    if points_norm.shape[0] == 0:
        return 0.0
    rs = np.random.RandomState(rng_seed)
    U = rs.rand(n_samples, 3)
    dominated = 0
    P = np.clip(points_norm, 0.0, 1.0)
    for u in U:
        if np.any(np.all(P <= u, axis=1)):
            dominated += 1
    return dominated / float(n_samples)


def objective_space_variance(points_norm: np.ndarray) -> float:
    """
    Mean variance across 3 dims in normalized space.
    """
    if points_norm.shape[0] <= 1:
        return 0.0
    v = np.var(points_norm, axis=0, ddof=0)
    return float(np.mean(v))


def best_objectives(objs: np.ndarray) -> Tuple[float, float, float]:
    """
    Per-run "best" values from the final set (Pareto archive or top_pool):
      - best CT = min CT
      - best LoadSTD = min LoadSTD
      - best QLoss = min QLoss
    """
    if objs.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.min(objs[:, 0])), float(np.min(objs[:, 1])), float(np.min(objs[:, 2]))


# =========================
# 7) Experiment runner (20 runs, per-run shared normalization)
# =========================
def run_ablation_suite(n_runs: int = 20, base_seed: int = 20260118):
    methods = [
        IQEAConfig(name="FULL", pop_size=30, max_iter=100, n_obs_base=3,
                   use_extended_encoding=True, n_obs_fixed_one=False,
                   objective_mode="multi", update_mode="improved"),

        # Abl-1
        IQEAConfig(name="Abl-1(NoExtEncoding)", pop_size=30, max_iter=100, n_obs_base=3,
                   use_extended_encoding=False, n_obs_fixed_one=False,
                   objective_mode="multi", update_mode="improved"),

        # Abl-2 (requested: pop_size=90, n_obs=1)
        IQEAConfig(name="Abl-2(nObs1_pop90)", pop_size=120, max_iter=100, n_obs_base=3,
                   use_extended_encoding=True, n_obs_fixed_one=True,
                   objective_mode="multi", update_mode="improved"),

        # Abl-3a (weighted-sum)
        IQEAConfig(name="Abl-3a(WeightedSum)", pop_size=30, max_iter=100, n_obs_base=3,
                   use_extended_encoding=True, n_obs_fixed_one=False,
                   objective_mode="weighted_sum", update_mode="improved",
                   ws_w=(1/3, 1/3, 1/3)),

        # Abl-4 (classic update)
        IQEAConfig(name="Abl-4(ClassicUpdate)", pop_size=30, max_iter=100, n_obs_base=3,
                   use_extended_encoding=True, n_obs_fixed_one=False,
                   objective_mode="multi", update_mode="classic"),
    ]

    # store per method metrics across runs
    hv_rec: Dict[str, List[float]] = {m.name: [] for m in methods}
    var_rec: Dict[str, List[float]] = {m.name: [] for m in methods}
    ct_rec: Dict[str, List[float]] = {m.name: [] for m in methods}
    loadstd_rec: Dict[str, List[float]] = {m.name: [] for m in methods}
    qloss_rec: Dict[str, List[float]] = {m.name: [] for m in methods}

    for r in range(n_runs):
        seed = base_seed + 1000 * r

        # 1) run all methods in this run (so we can normalize with shared bounds)
        objs_by_method = {}
        all_objs = []
        for m in methods:
            per_method_seed = seed + stable_name_hash(m.name) % 100000
            objs = run_iqea(m, per_method_seed)
            objs_by_method[m.name] = objs
            if objs.shape[0] > 0:
                all_objs.append(objs)

        if len(all_objs) == 0:
            # all empty: record zeros/nans
            for m in methods:
                hv_rec[m.name].append(0.0)
                var_rec[m.name].append(0.0)
                ct_rec[m.name].append(float("nan"))
                loadstd_rec[m.name].append(float("nan"))
                qloss_rec[m.name].append(float("nan"))
            print(f"Run {r+1:02d}/{n_runs}: all empty")
            continue

        all_objs = np.vstack(all_objs)
        mins = all_objs.min(axis=0)
        maxs = all_objs.max(axis=0)

        # 2) compute metrics per method with shared normalization
        for m in methods:
            objs = objs_by_method[m.name]
            bct, bstd, bq = best_objectives(objs)
            ct_rec[m.name].append(bct)
            loadstd_rec[m.name].append(bstd)
            qloss_rec[m.name].append(bq)

            if objs.shape[0] == 0:
                hv_rec[m.name].append(0.0)
                var_rec[m.name].append(0.0)
                continue

            pts_n = normalize_by_bounds(objs, mins, maxs)
            hv = hv_monte_carlo_3d(pts_n, n_samples=m.hv_mc_samples, rng_seed=seed + 777)
            var = objective_space_variance(pts_n)

            hv_rec[m.name].append(hv)
            var_rec[m.name].append(var)

        print(f"Run {r+1:02d}/{n_runs}: done")

    # 3) summary (mean ± std)
    def mean_std_nan(x):
        x = np.array(x, dtype=float)
        x = x[~np.isnan(x)]
        if x.size == 0:
            return float("nan"), float("nan")
        return float(x.mean()), float(x.std(ddof=0))

    print("\n" + "=" * 160)
    print("Ablation Core Metrics (20 runs): mean ± std")
    print("  - HV: Monte Carlo HV in normalized [0,1]^3 (ref=(1,1,1), minimization)")
    print("  - ObjVar: mean variance across 3 dims in normalized space")
    print("  - Best CT/LoadSTD/QLoss: min over the final set returned by each method per run")
    print("=" * 160)

    header = (
        f"{'Method':<24} | "
        f"{'HV均值':>8} {'HV标准差':>10} | "
        f"{'目标空间方差均值':>14} {'目标空间方差标准差':>16} | "
        f"{'最优CT均值(s)':>14} {'最优CT标准差(s)':>16} | "
        f"{'最优LoadSTD均值(s)':>18} {'最优LoadSTD标准差(s)':>20} | "
        f"{'最优QLoss均值':>12} {'最优QLoss标准差':>14}"
    )
    print(header)
    print("-" * 160)

    for m in methods:
        hv_mu, hv_sd = mean_std_nan(hv_rec[m.name])
        v_mu, v_sd = mean_std_nan(var_rec[m.name])
        ct_mu, ct_sd = mean_std_nan(ct_rec[m.name])
        ls_mu, ls_sd = mean_std_nan(loadstd_rec[m.name])
        q_mu, q_sd = mean_std_nan(qloss_rec[m.name])

        print(
            f"{m.name:<24} | "
            f"{hv_mu:8.4f} {hv_sd:10.4f} | "
            f"{v_mu:14.6f} {v_sd:16.6f} | "
            f"{ct_mu:14.4f} {ct_sd:16.4f} | "
            f"{ls_mu:18.4f} {ls_sd:20.4f} | "
            f"{q_mu:12.6f} {q_sd:14.6f}"
        )

    print("=" * 160)

    return {
        "HV": hv_rec,
        "ObjVar": var_rec,
        "BestCT": ct_rec,
        "BestLoadSTD": loadstd_rec,
        "BestQLoss": qloss_rec,
    }


if __name__ == "__main__":
    # You can adjust n_runs / base_seed if needed.
    run_ablation_suite(n_runs=20, base_seed=20260118)
