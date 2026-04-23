import argparse
import csv
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

"""Three-objective MOPSO for welding-line comparison experiments."""

# =========================
# Problem definition
# =========================
NUM_TASKS = 24
NUM_STATIONS = 6
MAX_CYCLE_TIME = 600

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
QUAL_ALPHA = 0.2
QUAL_BETA = 0.5
QUAL_GAMMA = 0.3
TARGET_EXPANSION = 0.05
POSITIVE_WEIGHT = 1.5
NEGATIVE_WEIGHT = 0.8

MAX_ARCHIVE_SIZE = 30
DEDUP_BY_DECISION = True
SOFT_REPAIR_MAX_ITERS = 50
SOFT_REPAIR_TOPK_RANDOM = 0.40
SOFT_REPAIR_TOPK = 5
SOFT_REPAIR_ALLOW_MOVE_PRE = False
SOFT_REPAIR_LOCAL_JITTER = 0.05


# =========================
# Environment info
# =========================
def _try_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


def _safe_run_command(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
        return out.strip()
    except Exception:
        return None


def get_environment_info() -> Dict[str, str]:
    info: Dict[str, str] = {
        "python_version": sys.version.replace("\n", " "),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "logical_cores": str(os.cpu_count()),
    }

    psutil = _try_import_psutil()
    if psutil is not None:
        try:
            info["physical_cores"] = str(psutil.cpu_count(logical=False))
        except Exception:
            info["physical_cores"] = "N/A"
        try:
            info["total_memory_gb"] = str(round(psutil.virtual_memory().total / (1024 ** 3), 4))
        except Exception:
            info["total_memory_gb"] = "N/A"
    else:
        info["physical_cores"] = "N/A"
        info["total_memory_gb"] = "N/A"

    cpu_name = None
    if platform.system().lower() == "windows":
        cpu_name = _safe_run_command(["wmic", "cpu", "get", "name"])
    elif platform.system().lower() == "linux":
        cpu_name = _safe_run_command(["bash", "-lc", "grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2"])
    elif platform.system().lower() == "darwin":
        cpu_name = _safe_run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
    if cpu_name:
        info["cpu_name"] = " ".join([x for x in cpu_name.splitlines() if x.strip() and "Name" not in x]).strip()
    else:
        info["cpu_name"] = "N/A"

    gpu_info = _safe_run_command([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"
    ])
    info["gpu_info"] = gpu_info if gpu_info else "N/A"
    return info


# =========================
# Core objective functions
# =========================
def simple_quality_model(task: int, station: int, tool: int):
    tool_base = {0: (0.8, 0.06, 0.02), 1: (0.6, 0.08, -0.01), 2: (0.5, 0.10, 0.03)}
    rb, db, eb = tool_base.get(tool, (0.7, 0.08, 0.0))
    task_scale = 1.0 + (TASK_TIMES[task] / (max(TASK_TIMES) + EPS)) * 0.2
    station_adjust = 1.0 + ((station % 3) - 1) * 0.2
    roughness = rb * task_scale * station_adjust
    defect_rate = min(max(db * task_scale * station_adjust, 0.0), 0.5)
    expansion = eb * task_scale * station_adjust
    return roughness, defect_rate, expansion


def calculate_quality_loss(station_assignment: np.ndarray, tool_assignment: np.ndarray) -> float:
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


def build_station_sequences_from_assignment(station_assignment: np.ndarray) -> List[List[int]]:
    station_sequences = [[] for _ in range(NUM_STATIONS)]
    for task, station in enumerate(station_assignment):
        station_sequences[int(station)].append(task)
    for s in range(NUM_STATIONS):
        random.shuffle(station_sequences[s])
    return station_sequences


def repair_tool_assignment(tool_assignment: np.ndarray) -> np.ndarray:
    ta = tool_assignment.copy()
    for i in range(NUM_TASKS):
        if int(ta[i]) not in ALLOWED_TOOLS[i]:
            ta[i] = random.choice(list(ALLOWED_TOOLS[i]))
    return ta


def _station_loads_proxy(sa: np.ndarray) -> np.ndarray:
    loads = np.zeros(NUM_STATIONS, dtype=float)
    for t in range(NUM_TASKS):
        loads[int(sa[t])] += TASK_TIMES[t]
    return loads


def _can_move_without_empty(sa: np.ndarray, t: int, s_from: int) -> bool:
    return int(np.sum(sa == s_from)) > 1


def _apply_move(sa: np.ndarray, t: int, s_to: int) -> np.ndarray:
    out = sa.copy()
    out[t] = int(s_to)
    return out


def repair_station_assignment_precedence_soft(station_assignment: np.ndarray) -> np.ndarray:
    sa = station_assignment.copy().astype(int)
    sa = np.clip(sa, 0, NUM_STATIONS - 1)

    if random.random() < SOFT_REPAIR_LOCAL_JITTER:
        t = random.randrange(NUM_TASKS)
        sa[t] = int(np.clip(sa[t] + random.choice([-1, 1]), 0, NUM_STATIONS - 1))

    for _ in range(SOFT_REPAIR_MAX_ITERS):
        violations = [(pre, post) for (pre, post) in PRECEDENCE_CONSTRAINTS if sa[pre] > sa[post]]
        if not violations:
            break

        pre, post = random.choice(violations)
        s_pre = int(sa[pre])
        s_post = int(sa[post])
        loads = _station_loads_proxy(sa)
        candidates = []

        for s_to in range(s_pre, NUM_STATIONS):
            if s_to == s_post:
                continue
            if not _can_move_without_empty(sa, post, s_post):
                break
            new_load_from = loads[s_post] - TASK_TIMES[post]
            new_load_to = loads[s_to] + TASK_TIMES[post]
            new_max = max(max(loads[j] for j in range(NUM_STATIONS) if j not in (s_post, s_to)), new_load_from, new_load_to)
            move_dist = abs(s_to - s_post)
            candidates.append(("move_post", post, s_to, float(new_max), move_dist))

        if SOFT_REPAIR_ALLOW_MOVE_PRE:
            for s_to in range(0, s_post + 1):
                if s_to == s_pre:
                    continue
                if not _can_move_without_empty(sa, pre, s_pre):
                    break
                new_load_from = loads[s_pre] - TASK_TIMES[pre]
                new_load_to = loads[s_to] + TASK_TIMES[pre]
                new_max = max(max(loads[j] for j in range(NUM_STATIONS) if j not in (s_pre, s_to)), new_load_from, new_load_to)
                move_dist = abs(s_to - s_pre)
                candidates.append(("move_pre", pre, s_to, float(new_max), move_dist))

        if not candidates:
            if _can_move_without_empty(sa, post, s_post):
                sa[post] = s_pre
            else:
                sa[post] = min(s_pre + 1, NUM_STATIONS - 1)
            continue

        candidates.sort(key=lambda x: (x[3], x[4]))
        k = min(SOFT_REPAIR_TOPK, len(candidates))
        pick = random.choice(candidates[:k]) if random.random() < SOFT_REPAIR_TOPK_RANDOM and k > 1 else candidates[0]
        _, task_id, s_to, _, _ = pick
        sa = _apply_move(sa, task_id, s_to)
        sa = np.clip(sa, 0, NUM_STATIONS - 1)

    return sa


def repair_no_empty_stations(station_assignment: np.ndarray) -> np.ndarray:
    sa = station_assignment.copy().astype(int)
    counts = np.bincount(sa, minlength=NUM_STATIONS)
    empty = [s for s in range(NUM_STATIONS) if counts[s] == 0]
    if not empty:
        return sa

    station_tasks = {s: np.where(sa == s)[0].tolist() for s in range(NUM_STATIONS)}
    for es in empty:
        donors = [s for s in range(NUM_STATIONS) if len(station_tasks[s]) > 1]
        if not donors:
            break
        donor = max(donors, key=lambda s: len(station_tasks[s]))
        task_to_move = random.choice(station_tasks[donor])
        sa[task_to_move] = es
        station_tasks[donor].remove(task_to_move)
        station_tasks[es].append(task_to_move)
    return sa


def repair_station_sequences_intra_precedence(station_assignment: np.ndarray, station_sequences: List[List[int]]) -> List[List[int]]:
    succs = {t: [] for t in range(NUM_TASKS)}
    for pre, post in PRECEDENCE_CONSTRAINTS:
        succs[pre].append(post)

    repaired: List[List[int]] = []
    for s in range(NUM_STATIONS):
        seq = station_sequences[s]
        if len(seq) <= 1:
            repaired.append(seq[:])
            continue

        set_tasks = set(seq)
        in_deg = {t: 0 for t in set_tasks}
        local_succ = {t: [] for t in set_tasks}
        for t in set_tasks:
            for u in succs[t]:
                if u in set_tasks:
                    local_succ[t].append(u)
        for t in set_tasks:
            for u in local_succ[t]:
                in_deg[u] += 1

        pos = {t: i for i, t in enumerate(seq)}
        ready = [t for t in set_tasks if in_deg[t] == 0]
        out: List[int] = []
        while ready:
            ready.sort(key=lambda x: pos.get(x, 10**9))
            t = ready.pop(0)
            out.append(t)
            for u in local_succ[t]:
                in_deg[u] -= 1
                if in_deg[u] == 0:
                    ready.append(u)
        repaired.append(out if len(out) == len(seq) else seq[:])
    return repaired


def compute_station_times_from_sequences(station_sequences: List[List[int]], tool_assignment: np.ndarray) -> np.ndarray:
    station_times = np.zeros(NUM_STATIONS, dtype=float)
    for s in range(NUM_STATIONS):
        seq = station_sequences[s]
        if not seq:
            continue
        station_times[s] += TASK_TIMES[seq[0]]
        prev_tool = int(tool_assignment[seq[0]])
        for t in seq[1:]:
            curr_tool = int(tool_assignment[t])
            station_times[s] += TASK_TIMES[t]
            if curr_tool != prev_tool:
                station_times[s] += TOOL_SWITCH_COST_MATRIX[prev_tool, curr_tool]
            prev_tool = curr_tool
    return station_times


def constraint_violation(station_assignment: np.ndarray, station_sequences: List[List[int]], cycle_time: float) -> float:
    v = 0.0
    empties = sum(1 for s in range(NUM_STATIONS) if len(station_sequences[s]) == 0)
    v += 15 * float(empties)

    for pre, post in PRECEDENCE_CONSTRAINTS:
        diff = int(station_assignment[pre]) - int(station_assignment[post])
        if diff > 0:
            v += 10 * float(diff)

    pos_in_station = {}
    for s in range(NUM_STATIONS):
        for idx, t in enumerate(station_sequences[s]):
            pos_in_station[t] = (s, idx)

    for pre, post in PRECEDENCE_CONSTRAINTS:
        sp = int(station_assignment[pre])
        sq = int(station_assignment[post])
        if sp == sq:
            if pre in pos_in_station and post in pos_in_station:
                _, ip = pos_in_station[pre]
                _, iq = pos_in_station[post]
                if ip > iq:
                    v += 5.0
            else:
                v += 10.0

    if MAX_CYCLE_TIME is not None:
        v += max(0.0, float(cycle_time) - float(MAX_CYCLE_TIME))
    return v


# =========================
# Particle and MOPSO
# =========================
class Particle:
    def __init__(self, station_assignment=None, tool_assignment=None, station_sequences=None):
        if station_assignment is None:
            base = [s for s in range(NUM_STATIONS)]
            rest = [random.randint(0, NUM_STATIONS - 1) for _ in range(NUM_TASKS - NUM_STATIONS)]
            sa = np.array(base + rest, dtype=int)
            np.random.shuffle(sa)
            self.station_assignment = sa
        else:
            self.station_assignment = np.array(station_assignment, dtype=int)

        if tool_assignment is None:
            self.tool_assignment = np.array([random.choice(list(ALLOWED_TOOLS[i])) for i in range(NUM_TASKS)], dtype=int)
        else:
            self.tool_assignment = np.array(tool_assignment, dtype=int)

        if station_sequences is None:
            self.station_sequences = build_station_sequences_from_assignment(self.station_assignment)
        else:
            self.station_sequences = [list(seq) for seq in station_sequences]

        self.velocity_station = np.random.uniform(-1, 1, NUM_TASKS)
        self.velocity_tool = np.random.uniform(-0.5, 0.5, NUM_TASKS)
        self.objectives = None
        self.constraint_violation = None
        self.feasible = None
        self.pbest_station = self.station_assignment.copy()
        self.pbest_tool = self.tool_assignment.copy()
        self.pbest_sequences = [seq.copy() for seq in self.station_sequences]
        self.pbest_objectives = None
        self.pbest_violation = None
        self.pbest_feasible = None
        self.domination_count = 0
        self.dominated_solutions = []

    def clone_shallow(self):
        p = Particle(self.station_assignment.copy(), self.tool_assignment.copy(), [seq.copy() for seq in self.station_sequences])
        p.objectives = None if self.objectives is None else self.objectives.copy()
        p.constraint_violation = self.constraint_violation
        p.feasible = self.feasible
        p.velocity_station = self.velocity_station.copy()
        p.velocity_tool = self.velocity_tool.copy()
        p.pbest_station = self.pbest_station.copy()
        p.pbest_tool = self.pbest_tool.copy()
        p.pbest_sequences = [seq.copy() for seq in self.pbest_sequences]
        p.pbest_objectives = None if self.pbest_objectives is None else self.pbest_objectives.copy()
        p.pbest_violation = self.pbest_violation
        p.pbest_feasible = self.pbest_feasible
        return p


class MOPSO_WeldingLine:
    def __init__(self, pop_size=30, max_gen=100, c1=1.2, c2=1.2, w=0.5, verbose_every=10):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.c1 = c1
        self.c2 = c2
        self.w = float(w)
        self.verbose_every = verbose_every
        self.archive: List[Particle] = []
        self.objective_evaluations = 0
        self.evaluation_wall_time_sec = 0.0
        self.total_wall_time_sec = 0.0

    def evaluate(self, particle: Particle):
        eval_t0 = time.perf_counter()
        particle.station_assignment = repair_station_assignment_precedence_soft(particle.station_assignment)
        particle.station_assignment = repair_no_empty_stations(particle.station_assignment)
        particle.tool_assignment = repair_tool_assignment(particle.tool_assignment)
        particle.station_sequences = build_station_sequences_from_assignment(particle.station_assignment)
        particle.station_sequences = repair_station_sequences_intra_precedence(particle.station_assignment, particle.station_sequences)

        station_times = compute_station_times_from_sequences(particle.station_sequences, particle.tool_assignment)
        ct = float(np.max(station_times))
        valid_station_times = station_times[station_times > 0]
        std_time = float(np.std(valid_station_times)) if len(valid_station_times) > 1 else 0.0
        qloss = float(calculate_quality_loss(particle.station_assignment, particle.tool_assignment))
        particle.objectives = np.array([ct, std_time, qloss], dtype=float)
        particle.constraint_violation = float(constraint_violation(particle.station_assignment, particle.station_sequences, ct))
        particle.feasible = particle.constraint_violation <= EPS
        self.objective_evaluations += 1
        self.evaluation_wall_time_sec += time.perf_counter() - eval_t0
        return particle.objectives

    def constrained_dominates(self, p: Particle, q: Particle) -> bool:
        if p.feasible and not q.feasible:
            return True
        if not p.feasible and q.feasible:
            return False
        if not p.feasible and not q.feasible:
            return p.constraint_violation < q.constraint_violation - EPS
        return np.all(p.objectives <= q.objectives + EPS) and np.any(p.objectives < q.objectives - EPS)

    def fast_non_dominated_sort(self, population: List[Particle]) -> List[List[Particle]]:
        fronts = [[]]
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []
            for q in population:
                if p is q:
                    continue
                if self.constrained_dominates(p, q):
                    p.dominated_solutions.append(q)
                elif self.constrained_dominates(q, p):
                    p.domination_count += 1
            if p.domination_count == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts[:-1]

    def _dedup_archive(self, particles: List[Particle]) -> List[Particle]:
        if not DEDUP_BY_DECISION:
            return particles
        uniq = []
        seen = set()
        for p in particles:
            key = (tuple(p.station_assignment.tolist()), tuple(p.tool_assignment.tolist()))
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        return uniq

    def update_archive(self, population: List[Particle]) -> None:
        combined = self.archive + [p.clone_shallow() for p in population]
        fronts = self.fast_non_dominated_sort(combined)
        front0 = fronts[0] if fronts else []
        front0 = self._dedup_archive(front0)
        if len(front0) > MAX_ARCHIVE_SIZE:
            indices = list(range(len(front0)))
            random.shuffle(indices)
            front0 = [front0[i] for i in indices[:MAX_ARCHIVE_SIZE]]
        self.archive = front0

    def select_gbest(self) -> Particle:
        if not self.archive:
            return Particle()
        return random.choice(self.archive)

    def pbest_better(self, obj_new, v_new, feas_new, obj_old, v_old, feas_old) -> bool:
        if feas_new and not feas_old:
            return True
        if not feas_new and feas_old:
            return False
        if not feas_new and not feas_old:
            return v_new < v_old - EPS
        return bool(np.all(obj_new <= obj_old + EPS) and np.any(obj_new < obj_old - EPS))

    def update_velocity_position(self, particle: Particle, gbest: Particle) -> None:
        r1, r2 = np.random.rand(), np.random.rand()
        vel_station_cognitive = self.c1 * r1 * (particle.pbest_station - particle.station_assignment)
        vel_station_social = self.c2 * r2 * (gbest.station_assignment - particle.station_assignment)
        particle.velocity_station = self.w * particle.velocity_station + vel_station_cognitive + vel_station_social
        for i in range(NUM_TASKS):
            if abs(particle.velocity_station[i]) > 0.3:
                step = int(np.sign(particle.velocity_station[i]))
                particle.station_assignment[i] = int(np.clip(particle.station_assignment[i] + step, 0, NUM_STATIONS - 1))

        vel_tool_cognitive = self.c1 * r1 * (particle.pbest_tool - particle.tool_assignment)
        vel_tool_social = self.c2 * r2 * (gbest.tool_assignment - particle.tool_assignment)
        particle.velocity_tool = self.w * particle.velocity_tool + vel_tool_cognitive + vel_tool_social
        for i in range(NUM_TASKS):
            if abs(particle.velocity_tool[i]) > 0.3:
                allowed = list(ALLOWED_TOOLS[i])
                if len(allowed) > 1:
                    cur = int(particle.tool_assignment[i])
                    cand = [t for t in allowed if t != cur]
                    if cand:
                        particle.tool_assignment[i] = random.choice(cand)

        particle.station_assignment = repair_station_assignment_precedence_soft(particle.station_assignment)
        particle.station_assignment = repair_no_empty_stations(particle.station_assignment)
        particle.tool_assignment = repair_tool_assignment(particle.tool_assignment)
        particle.station_sequences = build_station_sequences_from_assignment(particle.station_assignment)
        particle.station_sequences = repair_station_sequences_intra_precedence(particle.station_assignment, particle.station_sequences)

        if random.random() < 0.2:
            s = random.randint(0, NUM_STATIONS - 1)
            seq = particle.station_sequences[s]
            if len(seq) > 2:
                i1, i2 = random.sample(range(len(seq)), 2)
                seq[i1], seq[i2] = seq[i2], seq[i1]
                particle.station_sequences = repair_station_sequences_intra_precedence(particle.station_assignment, particle.station_sequences)

    def evolve(self, verbose=False):
        run_t0 = time.perf_counter()
        population = [Particle() for _ in range(self.pop_size)]
        for p in population:
            self.evaluate(p)
            p.pbest_objectives = p.objectives.copy()
            p.pbest_violation = p.constraint_violation
            p.pbest_feasible = p.feasible
        self.update_archive(population)

        for gen in range(self.max_gen):
            for p in population:
                gbest = self.select_gbest()
                self.update_velocity_position(p, gbest)
                self.evaluate(p)
                if self.pbest_better(p.objectives, p.constraint_violation, p.feasible, p.pbest_objectives, p.pbest_violation, p.pbest_feasible):
                    p.pbest_station = p.station_assignment.copy()
                    p.pbest_tool = p.tool_assignment.copy()
                    p.pbest_sequences = [seq.copy() for seq in p.station_sequences]
                    p.pbest_objectives = p.objectives.copy()
                    p.pbest_violation = p.constraint_violation
                    p.pbest_feasible = p.feasible
            self.update_archive(population)
            if verbose and ((gen == 0) or ((gen + 1) % self.verbose_every == 0) or (gen + 1 == self.max_gen)):
                feas_archive = [p for p in self.archive if p.feasible]
                if feas_archive:
                    objs = np.array([p.objectives for p in feas_archive], dtype=float)
                    print(
                        f"MOPSO Gen {gen + 1:3d}/{self.max_gen:3d} | Archive={len(feas_archive):3d} | "
                        f"Best CT={objs[:,0].min():8.2f} | Best LoadSTD={objs[:,1].min():8.2f} | Best QLoss={objs[:,2].min():10.4f}"
                    )
        self.total_wall_time_sec = time.perf_counter() - run_t0
        return self.archive


# =========================
# Front utilities and metrics
# =========================
def particle_to_solution(p: Particle):
    return (p.station_assignment.copy(), [seq.copy() for seq in p.station_sequences], p.tool_assignment.copy()), tuple(float(x) for x in p.objectives.tolist())


def front_to_array(front) -> np.ndarray:
    if not front:
        return np.empty((0, 3), dtype=float)
    return np.asarray([obj for _, obj in front], dtype=float)


def select_representative_by_ideal(front):
    if not front:
        return None
    objs = np.array([obj for _, obj in front], dtype=float)
    ideal = objs.min(axis=0)
    nadir = objs.max(axis=0)
    denom = np.maximum(nadir - ideal, EPS)
    norm = (objs - ideal) / denom
    d = np.sqrt(np.sum(norm * norm, axis=1))
    idx = int(np.argmin(d))
    return front[idx], ideal


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
    return np.unique(points[keep], axis=0)


def get_reference_point(reference_front: np.ndarray, approx_front: np.ndarray) -> np.ndarray:
    combined = reference_front if approx_front.size == 0 else np.vstack([reference_front, approx_front])
    mins = np.min(combined, axis=0)
    maxs = np.max(combined, axis=0)
    ranges = np.maximum(maxs - mins, 1.0)
    return maxs + 0.1 * ranges


def compute_hypervolume_3d(points: np.ndarray, ref_point: np.ndarray) -> float:
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
        dists.append(np.min(np.linalg.norm(approx_norm - rp, axis=1)))
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
        w.writerow(["run", "seed", "CT", "LoadSTD", "QLoss", "HV", "IGD", "runtime_sec", "n_obj_eval_total"])
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
    parser = argparse.ArgumentParser(description="MOPSO comparison script for review reproducibility.")
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-gen", type=int, default=100)
    parser.add_argument("--c1", type=float, default=1.2)
    parser.add_argument("--c2", type=float, default=1.2)
    parser.add_argument("--w", type=float, default=0.5)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--out-dir", type=str, default="results/mopso")
    parser.add_argument("--reference-front", type=str, default=None, help="CSV file with columns CT,LoadSTD,QLoss. Required for HV/IGD.")
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
        print(f"MOPSO run {run:02d}/{args.num_runs:02d} starts...")

        mopso = MOPSO_WeldingLine(pop_size=args.pop_size, max_gen=args.max_gen, c1=args.c1, c2=args.c2, w=args.w)
        archive = mopso.evolve(verbose=args.verbose)
        final_pareto_front = [particle_to_solution(p) for p in archive if p.feasible]
        front_csv = save_front_csv(final_pareto_front, run, args.out_dir)

        if final_pareto_front:
            rep_pack = select_representative_by_ideal(final_pareto_front)
            (final_sol, final_obj), _ideal = rep_pack
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
                f"HV={hv:.6f}, IGD={igd:.6f}, Runtime={mopso.total_wall_time_sec:.4f}s, "
                f"ObjEval={mopso.objective_evaluations}, |front|={len(final_pareto_front)}"
            )
        else:
            ct_f, std_f, ql_f = np.nan, np.nan, np.nan
            hv, igd = np.nan, np.nan
            print(
                f"Run {run:02d}: final Pareto front empty | Runtime={mopso.total_wall_time_sec:.4f}s | "
                f"ObjEval={mopso.objective_evaluations}"
            )

        per_run_rows.append([
            run,
            seed,
            float(ct_f),
            float(std_f),
            float(ql_f),
            float(hv),
            float(igd),
            float(mopso.total_wall_time_sec),
            float(mopso.objective_evaluations),
        ])
        print(f"Saved front: {front_csv}")

    per_run_csv = save_per_run_metrics(per_run_rows, args.out_dir)
    summary_csv = save_summary_statistics(build_summary_rows(per_run_rows), args.out_dir)

    print("\n" + "=" * 90)
    print("MOPSO comparison experiment completed.")
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
