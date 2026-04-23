import argparse
import csv
import json
import os
import platform
import random
import socket
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

"""Three-objective NSGA-II for welding-line comparison experiments."""

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
MAX_ARCHIVE_SIZE = 20
QUAL_ALPHA = 0.2
QUAL_BETA = 0.5
QUAL_GAMMA = 0.3
TARGET_EXPANSION = 0.05
POSITIVE_WEIGHT = 1.5
NEGATIVE_WEIGHT = 0.8
EPS = 1e-12


# =========================
# Environment info
# =========================
def get_environment_info() -> Dict[str, str]:
    info: Dict[str, str] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "system_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() if platform.processor() else "N/A",
        "python_version": sys.version.replace("\n", " "),
        "numpy_version": np.__version__,
        "cpu_logical_count": str(os.cpu_count()),
    }
    try:
        import psutil  # type: ignore
        info["physical_cpu_count"] = str(psutil.cpu_count(logical=False))
        info["total_memory_gb"] = str(round(psutil.virtual_memory().total / (1024 ** 3), 3))
    except Exception:
        info["physical_cpu_count"] = "N/A"
        info["total_memory_gb"] = "N/A"
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


def compute_station_times(solution):
    station_assignment, station_sequences, tool_assignment = solution
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


def compute_crowding_distance_from_objs(objs: np.ndarray) -> np.ndarray:
    n = objs.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    if n <= 2:
        return np.full(n, np.inf, dtype=float)
    cd = np.zeros(n, dtype=float)
    m = objs.shape[1]
    for k in range(m):
        idx = np.argsort(objs[:, k])
        cd[idx[0]] = np.inf
        cd[idx[-1]] = np.inf
        fmin = objs[idx[0], k]
        fmax = objs[idx[-1], k]
        denom = fmax - fmin
        if denom <= EPS:
            continue
        for j in range(1, n - 1):
            if np.isinf(cd[idx[j]]):
                continue
            cd[idx[j]] += (objs[idx[j + 1], k] - objs[idx[j - 1], k]) / denom
    return cd


def dominates_simplified(a, b):
    return all(a[i] <= b[i] + EPS for i in range(3)) and any(a[i] < b[i] - EPS for i in range(3))


def solution_key(solution):
    station_assignment, station_sequences, tool_selection = solution
    return (
        tuple(station_assignment.tolist()),
        tuple(tool_selection.tolist()),
        tuple(tuple(seq) for seq in station_sequences),
    )


def update_archive_simplified(archive, cand_solution, cand_obj, obj_tol=1e-9):
    ck = solution_key(cand_solution)
    for sol, obj in archive:
        if solution_key(sol) == ck:
            return archive
    for sol, obj in archive:
        if all(abs(obj[i] - cand_obj[i]) <= obj_tol for i in range(3)):
            return archive
    for _, obj in archive:
        if dominates_simplified(obj, cand_obj):
            return archive
    new_archive = [(sol, obj) for sol, obj in archive if not dominates_simplified(cand_obj, obj)]
    new_archive.append((cand_solution, cand_obj))
    if len(new_archive) > MAX_ARCHIVE_SIZE:
        objs = np.array([obj for _, obj in new_archive], dtype=float)
        cd = compute_crowding_distance_from_objs(objs)
        keep_idx = np.argsort(cd)[::-1][:MAX_ARCHIVE_SIZE]
        new_archive = [new_archive[i] for i in keep_idx]
    return new_archive


# =========================
# Individual and NSGA-II
# =========================
class Individual:
    def __init__(self, task_sequence: List[int], station_assignment: List[int], tool_selection: List[int]):
        self.task_sequence = task_sequence
        self.station_assignment = station_assignment
        self.tool_selection = tool_selection
        self.objectives = None
        self.objectives_raw = None
        self.violation = 0.0
        self.rank = None
        self.crowding_distance = 0.0
        self.domination_count = 0
        self.dominated_solutions: List[Individual] = []

    def to_tuple(self):
        station_sequences = [[] for _ in range(NUM_STATIONS)]
        for task in self.task_sequence:
            station = self.station_assignment[task]
            station_sequences[station].append(task)
        return (
            np.array(self.station_assignment, dtype=int),
            [list(seq) for seq in station_sequences],
            np.array(self.tool_selection, dtype=int),
        )


class Simple_NSGA2_WeldingLine:
    def __init__(self, pop_size=30, max_gen=100, cx_prob=0.6, mut_prob=0.2):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.archive = []
        self.eval_count = 0
        self.objective_eval_time = 0.0
        self.last_run_wall_clock = 0.0

    def generate_individual(self) -> Individual:
        task_sequence = list(range(NUM_TASKS))
        random.shuffle(task_sequence)
        station_assignment = [random.randint(0, NUM_STATIONS - 1) for _ in range(NUM_TASKS)]
        tool_selection = [random.choice(list(ALLOWED_TOOLS[i])) for i in range(NUM_TASKS)]
        individual = Individual(task_sequence, station_assignment, tool_selection)
        individual.objectives = self.evaluate(individual)
        return individual

    def evaluate(self, individual: Individual) -> List[float]:
        eval_t0 = time.perf_counter()
        solution = individual.to_tuple()
        station_times = compute_station_times(solution)
        cycle_time = float(np.max(station_times))
        valid_station_times = station_times[station_times > 0]
        load_std = float(np.std(valid_station_times)) if len(valid_station_times) > 1 else 0.0
        quality_loss = float(calculate_quality_loss(solution))
        individual.objectives_raw = [cycle_time, load_std, quality_loss]
        station_counts = [0] * NUM_STATIONS
        for task in individual.task_sequence:
            station_counts[individual.station_assignment[task]] += 1
        v_empty = sum(1 for count in station_counts if count == 0) * 50.0
        v_ct = max(0.0, cycle_time - MAX_CYCLE_TIME) * 10.0
        individual.violation = v_ct + v_empty
        penalty = v_ct + v_empty * 5.0
        self.eval_count += 1
        self.objective_eval_time += time.perf_counter() - eval_t0
        return [cycle_time + penalty, load_std + penalty, quality_loss + penalty]

    def simple_dominates(self, ind1: Individual, ind2: Individual) -> bool:
        obj1, obj2 = ind1.objectives, ind2.objectives
        return all(obj1[i] <= obj2[i] + EPS for i in range(3)) and any(obj1[i] < obj2[i] - EPS for i in range(3))

    def fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        fronts = [[]]
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []
            for q in population:
                if p is q:
                    continue
                if self.simple_dominates(p, q):
                    p.dominated_solutions.append(q)
                elif self.simple_dominates(q, p):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 1
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 2
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts[:-1]

    def calculate_crowding_distance(self, front: List[Individual]) -> None:
        if len(front) == 0:
            return
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float("inf")
            return
        objs = np.array([ind.objectives for ind in front], dtype=float)
        cd = compute_crowding_distance_from_objs(objs)
        for ind, d in zip(front, cd):
            ind.crowding_distance = float(d)

    def tournament_selection(self, population: List[Individual]) -> Individual:
        tournament = random.sample(population, 3)
        best = tournament[0]
        for ind in tournament[1:]:
            if ind.rank < best.rank:
                best = ind
            elif ind.rank == best.rank and ind.crowding_distance > best.crowding_distance:
                best = ind
        return best

    def _repair_sequence_simple(self, sequence: List[int]) -> List[int]:
        seen = set()
        result = []
        for task in sequence:
            if task not in seen:
                result.append(task)
                seen.add(task)
        for task in range(NUM_TASKS):
            if task not in seen:
                result.append(task)
        return result

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if random.random() > self.cx_prob:
            return parent1, parent2
        point = random.randint(1, NUM_TASKS - 2)
        child1_seq = self._repair_sequence_simple(parent1.task_sequence[:point] + parent2.task_sequence[point:])
        child2_seq = self._repair_sequence_simple(parent2.task_sequence[:point] + parent1.task_sequence[point:])

        child1_station, child2_station = [], []
        child1_tool, child2_tool = [], []
        for i in range(NUM_TASKS):
            if random.random() < 0.5:
                child1_station.append(parent1.station_assignment[i])
                child2_station.append(parent2.station_assignment[i])
                child1_tool.append(parent1.tool_selection[i])
                child2_tool.append(parent2.tool_selection[i])
            else:
                child1_station.append(parent2.station_assignment[i])
                child2_station.append(parent1.station_assignment[i])
                child1_tool.append(parent2.tool_selection[i])
                child2_tool.append(parent1.tool_selection[i])
        child1 = Individual(child1_seq, child1_station, child1_tool)
        child2 = Individual(child2_seq, child2_station, child2_tool)
        child1.objectives = self.evaluate(child1)
        child2.objectives = self.evaluate(child2)
        return child1, child2

    def mutation(self, individual: Individual) -> Individual:
        if random.random() > self.mut_prob:
            return individual
        mutated = Individual(individual.task_sequence.copy(), individual.station_assignment.copy(), individual.tool_selection.copy())
        if random.random() < 0.3 and len(mutated.task_sequence) >= 2:
            i, j = random.sample(range(len(mutated.task_sequence)), 2)
            mutated.task_sequence[i], mutated.task_sequence[j] = mutated.task_sequence[j], mutated.task_sequence[i]
        mutated.objectives = self.evaluate(mutated)
        return mutated

    def evolve(self, verbose=False):
        run_t0 = time.perf_counter()
        population = [self.generate_individual() for _ in range(self.pop_size)]
        for gen in range(self.max_gen):
            fronts_pop = self.fast_non_dominated_sort(population)
            for front in fronts_pop:
                self.calculate_crowding_distance(front)

            offspring: List[Individual] = []
            while len(offspring) < self.pop_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                offspring.extend([child1, child2])
            offspring = offspring[: self.pop_size]

            combined = population + offspring
            fronts = self.fast_non_dominated_sort(combined)
            for front in fronts:
                self.calculate_crowding_distance(front)

            new_population: List[Individual] = []
            for front in fronts:
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend(front)
                else:
                    remaining = self.pop_size - len(new_population)
                    new_population.extend(front[:remaining])
                    break
            population = new_population

            for ind in population:
                if ind.violation <= 8.0:
                    solution = ind.to_tuple()
                    obj_raw = tuple(ind.objectives_raw)
                    self.archive = update_archive_simplified(self.archive, solution, obj_raw)

            if verbose and (gen == 0 or gen % 20 == 0 or gen == self.max_gen - 1):
                if self.archive:
                    archive_objs = np.array([obj for _, obj in self.archive], dtype=float)
                    print(
                        f"Gen {gen:3d}/{self.max_gen:3d} | Archive={len(self.archive):2d} | "
                        f"BestCT={archive_objs[:,0].min():8.2f} | Evals={self.eval_count:5d}"
                    )
                else:
                    print(f"Gen {gen:3d}/{self.max_gen:3d} | Archive= 0 | BestCT=N/A | Evals={self.eval_count:5d}")

        self.last_run_wall_clock = time.perf_counter() - run_t0
        return self.archive


# =========================
# Front utilities and metrics
# =========================
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
    parser = argparse.ArgumentParser(description="NSGA-II comparison script for review reproducibility.")
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-gen", type=int, default=100)
    parser.add_argument("--cx-prob", type=float, default=0.6)
    parser.add_argument("--mut-prob", type=float, default=0.2)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--out-dir", type=str, default="results/nsgaii")
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
        print(f"NSGA-II run {run:02d}/{args.num_runs:02d} starts...")

        nsga2 = Simple_NSGA2_WeldingLine(pop_size=args.pop_size, max_gen=args.max_gen, cx_prob=args.cx_prob, mut_prob=args.mut_prob)
        archive = nsga2.evolve(verbose=args.verbose)
        final_pareto_front = list(archive)
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
                f"HV={hv:.6f}, IGD={igd:.6f}, Runtime={nsga2.last_run_wall_clock:.4f}s, "
                f"ObjEval={nsga2.eval_count}, |front|={len(final_pareto_front)}"
            )
        else:
            ct_f, std_f, ql_f = np.nan, np.nan, np.nan
            hv, igd = np.nan, np.nan
            print(
                f"Run {run:02d}: final Pareto front empty | Runtime={nsga2.last_run_wall_clock:.4f}s | "
                f"ObjEval={nsga2.eval_count}"
            )

        per_run_rows.append([
            run,
            seed,
            float(ct_f),
            float(std_f),
            float(ql_f),
            float(hv),
            float(igd),
            float(nsga2.last_run_wall_clock),
            float(nsga2.eval_count),
        ])
        print(f"Saved front: {front_csv}")

    per_run_csv = save_per_run_metrics(per_run_rows, args.out_dir)
    summary_csv = save_summary_statistics(build_summary_rows(per_run_rows), args.out_dir)

    print("\n" + "=" * 90)
    print("NSGA-II comparison experiment completed.")
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
