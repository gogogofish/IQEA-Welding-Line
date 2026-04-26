
"""
GitHub-ready combined reproducibility script.

Included algorithms:
1. runnable two-objective IQEA
2. runnable Double-population GA
3. runnable IPSO

Built-in benchmark instances:
- kilbridge45: 45 tasks, 10 stations
- weld24: 24 tasks, 6 stations

Examples
--------
Run IQEA on the 45-task instance:
    Benchmark Verification.py --algorithm iqea --instance kilbridge45 --seed 12345

Run Double GA on the 45-task instance:
    Benchmark Verification.py --algorithm doublega --instance kilbridge45 --seed 12345

Run IPSO on the 24-task instance:
    Benchmark Verification.py --algorithm ipso --instance weld24 --seed 12345
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Benchmark instances
# ============================================================================
def build_kilbridge45_instance():
    tasks = list(range(45))
    task_times = [
        9, 9, 10, 10, 17, 17, 13, 13, 20, 20, 10, 11, 6, 22, 11,
        19, 12, 3, 7, 4, 55, 14, 27, 29, 26, 6, 5, 24, 4, 5, 7,
        4, 15, 3, 7, 9, 4, 7, 5, 4, 21, 12, 6, 5, 5
    ]
    num_stations = 10
    precedence_relations = [
        (0, 2), (0, 6), (1, 3), (1, 7), (2, 4), (3, 5), (4, 8), (5, 9),
        (6, 8), (6, 13), (7, 9), (7, 13), (8, 40), (9, 40), (10, 12),
        (11, 12), (11, 36), (12, 13), (12, 14), (13, 16), (13, 24),
        (13, 28), (13, 29), (13, 30), (13, 31), (14, 15), (14, 17),
        (14, 22), (14, 23), (15, 18), (16, 25), (16, 26), (17, 18),
        (18, 19), (18, 32), (19, 20), (20, 21), (21, 27), (22, 32),
        (23, 32), (24, 25), (25, 37), (26, 27), (26, 32), (27, 37),
        (28, 40), (29, 40), (30, 40), (31, 40), (32, 33), (32, 34),
        (32, 35), (33, 37), (34, 39), (35, 37), (36, 42), (37, 39),
        (38, 40), (39, 40), (40, 41), (41, 43), (41, 44)
    ]
    precedence_matrix = np.zeros((45, 45), dtype=int)
    for i, j in precedence_relations:
        precedence_matrix[i, j] = 1
    return tasks, task_times, precedence_matrix, num_stations, precedence_relations


def build_weld24_instance():
    tasks = list(range(24))
    task_times = [
        55, 65, 55, 45, 45, 35, 55, 160, 35, 70, 100, 80,
        35, 60, 60, 160, 35, 70, 450, 40, 35, 420, 40, 35
    ]
    num_stations = 6
    precedence_relations = [
        (0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
        (9, 11), (10, 11), (11, 12), (13, 15), (14, 15), (15, 16),
        (17, 18), (18, 19), (8, 20), (12, 20), (16, 20), (19, 20),
        (20, 21), (21, 22), (22, 23)
    ]
    precedence_matrix = np.zeros((24, 24), dtype=int)
    for i, j in precedence_relations:
        precedence_matrix[i, j] = 1
    return tasks, task_times, precedence_matrix, num_stations, precedence_relations


def load_instance(instance_name: str):
    if instance_name == "kilbridge45":
        return build_kilbridge45_instance()
    if instance_name == "weld24":
        return build_weld24_instance()
    raise ValueError(f"Unsupported instance: {instance_name}")


def save_station_assignment_csv(path: Path, station_assignment, task_times):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["station_id", "station_time", "tasks_1based", "task_times"])
        for idx, station in enumerate(station_assignment, start=1):
            station_time = sum(task_times[t] for t in station)
            tasks_1based = [t + 1 for t in station]
            times = [task_times[t] for t in station]
            writer.writerow([idx, station_time, " ".join(map(str, tasks_1based)), " ".join(map(str, times))])


# ============================================================================
# runnable two-objective IQEA
# ============================================================================
class TwoObjectiveIQEA:
    def __init__(
        self,
        tasks: Sequence[int],
        task_times: Sequence[int],
        precedence_matrix: np.ndarray,
        num_stations: int,
        pop_size: int = 30,
        max_iter: int = 100,
        n_obs_base: int = 3,
        use_local_improve: bool = True,
        eps_ct: float = 0.0,
        eps_std: float = 0.0,
        max_archive_size: int = 20,
    ) -> None:
        self.tasks = list(tasks)
        self.task_times = list(task_times)
        self.precedence_matrix = precedence_matrix
        self.num_tasks = len(tasks)
        self.num_stations = num_stations

        self.pop_size = pop_size
        self.max_iter = max_iter
        self.n_obs_base = n_obs_base
        self.use_local_improve = use_local_improve
        self.eps_ct = eps_ct
        self.eps_std = eps_std
        self.max_archive_size = max_archive_size
        self.eps = 1e-12

        self.global_succ = [[] for _ in range(self.num_tasks)]
        self.global_pred = [[] for _ in range(self.num_tasks)]
        for u in range(self.num_tasks):
            for v in range(self.num_tasks):
                if self.precedence_matrix[u, v] == 1:
                    self.global_succ[u].append(v)
                    self.global_pred[v].append(u)

        self.archive = []
        self.history = {"best_ct": [], "best_load_std": [], "archive_size": []}

    def topo_sort_with_random_priority(self, tasks_in_station, rand_keys):
        if len(tasks_in_station) <= 1:
            return tasks_in_station[:]

        task_set = set(tasks_in_station)
        indeg = {t: 0 for t in tasks_in_station}
        succ = {t: [] for t in tasks_in_station}

        for u in tasks_in_station:
            for v in self.global_succ[u]:
                if v in task_set:
                    succ[u].append(v)
                    indeg[v] += 1

        available = [t for t in tasks_in_station if indeg[t] == 0]
        available.sort(key=lambda x: rand_keys[x])

        order = []
        while available:
            u = available.pop(0)
            order.append(u)
            for v in succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    available.append(v)
            available.sort(key=lambda x: rand_keys[x])

        if len(order) != len(tasks_in_station):
            return sorted(tasks_in_station, key=lambda x: rand_keys[x])
        return order

    def compute_station_times(self, solution):
        station_assignment, station_sequences = solution
        station_times = np.zeros(self.num_stations, dtype=float)
        for s in range(self.num_stations):
            seq = station_sequences[s]
            if not seq:
                continue
            station_times[s] = float(sum(self.task_times[t] for t in seq))
        return station_times

    def has_empty_station(self, solution) -> bool:
        _, station_sequences = solution
        return any(len(seq) == 0 for seq in station_sequences)

    def evaluate_objectives_with_penalty(self, solution):
        station_assignment, station_sequences = solution
        penalty = 0.0

        for pre, post in [(u, v) for u in range(self.num_tasks) for v in self.global_succ[u]]:
            sp = int(station_assignment[pre])
            sq = int(station_assignment[post])
            if sp > sq:
                penalty += 5.0
            elif sp == sq:
                seq = station_sequences[sp]
                try:
                    if seq.index(pre) > seq.index(post):
                        penalty += 3.0
                except ValueError:
                    penalty += 1.0

        station_times = self.compute_station_times(solution)
        ct = float(np.max(station_times)) if len(station_times) > 0 else 0.0
        load_std = float(np.std(station_times, ddof=0)) if self.num_stations > 1 else 0.0

        return (ct, load_std), (ct + penalty, load_std + penalty), penalty

    def dominates(self, a, b):
        ct_dom = a[0] <= b[0] + self.eps_ct
        std_dom = a[1] <= b[1] + self.eps_std
        all_dom = ct_dom and std_dom

        ct_strict = a[0] < b[0] - self.eps_ct
        std_strict = a[1] < b[1] - self.eps_std
        any_strict = ct_strict or std_strict

        return all_dom and any_strict

    def calculate_crowding_distance(self, pareto_front):
        n = len(pareto_front)
        if n == 0:
            return []

        objectives = np.array([obj for _, obj in pareto_front], dtype=float)
        crowd = np.zeros(n, dtype=float)

        for m in range(2):
            idx = np.argsort(objectives[:, m])
            crowd[idx[0]] = np.inf
            crowd[idx[-1]] = np.inf
            if n > 2:
                minv = objectives[idx[0], m]
                maxv = objectives[idx[-1], m]
                rng = maxv - minv
                if rng > self.eps:
                    for k in range(1, n - 1):
                        prevv = objectives[idx[k - 1], m]
                        nextv = objectives[idx[k + 1], m]
                        crowd[idx[k]] += (nextv - prevv) / rng
        return crowd.tolist()

    def update_archive(self, archive, cand_solution, cand_obj):
        for _, obj in archive:
            if self.dominates(obj, cand_obj):
                return archive

        new_archive = [(sol, obj) for sol, obj in archive if not self.dominates(cand_obj, obj)]
        new_archive.append((cand_solution, cand_obj))

        if len(new_archive) > self.max_archive_size:
            crowd = self.calculate_crowding_distance(new_archive)
            keep_idx = np.argsort(crowd)[::-1][: self.max_archive_size]
            new_archive = [new_archive[i] for i in keep_idx]
        return new_archive

    def select_representative_solutions(self, pareto_archive):
        reps = []
        if not pareto_archive:
            return reps

        objs = np.array([obj for _, obj in pareto_archive], dtype=float)
        mins = objs.min(axis=0)
        maxs = objs.max(axis=0)
        ranges = np.maximum(maxs - mins, self.eps)
        norm = (objs - mins) / ranges

        utopia = np.zeros(2)
        d_utopia = np.linalg.norm(norm - utopia, axis=1)
        idx_ideal = int(np.argmin(d_utopia))
        reps.append(("Ideal-point", pareto_archive[idx_ideal]))

        nadir = np.ones(2)
        line = (nadir - utopia)
        line = line / (np.linalg.norm(line) + self.eps)
        proj = (norm @ line.reshape(-1, 1)) * line.reshape(1, -1)
        perp = norm - proj
        dist_line = np.linalg.norm(perp, axis=1)
        idx_knee = int(np.argmax(dist_line))
        reps.append(("Knee-point", pareto_archive[idx_knee]))

        return reps

    def observe_individual(self, Q_j, n_obs, num_qubits):
        candidates = []
        for _ in range(n_obs):
            observed_bits = []
            ptr = 0
            station_assignment = np.zeros(self.num_tasks, dtype=int)
            rand_keys = np.zeros(self.num_tasks, dtype=float)

            for task in range(self.num_tasks):
                probs = []
                for s in range(self.num_stations):
                    alpha, beta = Q_j[ptr]
                    p0 = float(abs(alpha) ** 2)
                    probs.append(p0)
                    observed_bits.append(0 if np.random.rand() < p0 else 1)
                    ptr += 1
                probs = np.array(probs, dtype=float)
                probs = probs / probs.sum() if probs.sum() > self.eps else (np.ones(self.num_stations) / self.num_stations)
                station_assignment[task] = int(np.random.choice(self.num_stations, p=probs))

            for task in range(self.num_tasks):
                alpha, beta = Q_j[ptr]
                bit = 0 if np.random.rand() < float(abs(alpha) ** 2) else 1
                observed_bits.append(bit)
                base = np.random.rand()
                rand_keys[task] = 0.5 * base if bit == 0 else 0.5 + 0.5 * base
                ptr += 1

            station_sequences = [[] for _ in range(self.num_stations)]
            for task, s in enumerate(station_assignment):
                station_sequences[int(s)].append(int(task))
            for s in range(self.num_stations):
                station_sequences[s] = self.topo_sort_with_random_priority(station_sequences[s], rand_keys)

            sol = (station_assignment, station_sequences)
            orig_obj, pen_obj, _ = self.evaluate_objectives_with_penalty(sol)
            candidates.append((sol, np.array(observed_bits, dtype=int), orig_obj, pen_obj))

        dominated_counts = []
        for i, (_, _, _, pen_i) in enumerate(candidates):
            cnt = 0
            for j, (_, _, _, pen_j) in enumerate(candidates):
                if i != j and self.dominates(pen_j, pen_i):
                    cnt += 1
            dominated_counts.append(cnt)

        best_idx = int(np.argmin(dominated_counts))
        return candidates[best_idx][0], candidates[best_idx][1], candidates[best_idx][2]

    def update_Q(self, Q, guided_archive, X_obs, t, max_iter):
        pop_size, num_qubits, _ = Q.shape
        new_Q = np.copy(Q)
        valid_size = len(guided_archive)
        if valid_size == 0:
            return new_Q

        progress = t / max_iter
        reps = self.select_representative_solutions(guided_archive)
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
            ref_station = np.array([i % self.num_stations for i in range(self.num_tasks)], dtype=int)
            ref_seq = [[] for _ in range(self.num_stations)]
            for i, s in enumerate(ref_station):
                ref_seq[int(s)].append(int(i))
            ref_solution = (ref_station, ref_seq)

        ref_bits = []
        station_assignment, station_sequences = ref_solution
        for task in range(self.num_tasks):
            cs = int(station_assignment[task])
            for s in range(self.num_stations):
                ref_bits.append(1 if s == cs else 0)

        for task in range(self.num_tasks):
            s = int(station_assignment[task])
            seq = station_sequences[s]
            if task in seq and len(seq) > 0:
                rk = seq.index(task)
                ref_bits.append(1 if rk >= len(seq) / 2 else 0)
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
            j = int(update_indices[idx])
            for i in range(num_qubits):
                alpha, beta = Q[j, i]
                xi = int(X_obs[idx, i])
                ri = int(ref_bits[i])

                if xi == ri:
                    direction = 1.0 if np.random.rand() < 0.7 else -1.0
                else:
                    direction = 1.0 if ri == 1 else -1.0

                theta = base_angle * direction
                c, s = np.cos(theta), np.sin(theta)
                new_alpha = c * alpha - s * beta
                new_beta = s * alpha + c * beta

                norm = np.hypot(new_alpha, new_beta)
                if norm < self.eps:
                    new_alpha, new_beta = alpha, beta
                    norm = np.hypot(new_alpha, new_beta) + self.eps
                new_alpha /= norm
                new_beta /= norm

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
                    if norm2 < self.eps:
                        new_alpha = new_beta = 1 / np.sqrt(2)
                    else:
                        new_alpha /= norm2
                        new_beta /= norm2

                new_Q[j, i, 0] = new_alpha
                new_Q[j, i, 1] = new_beta

        return new_Q

    def local_2obj_improvement(self, solution, obj_values, max_tries=60):
        station_assignment, station_sequences = solution
        best_sol = solution
        best_obj = obj_values

        for _ in range(max_tries):
            move_type = "move" if np.random.rand() < 0.6 else "swap"
            new_station_assignment = station_assignment.copy()
            new_station_sequences = [seq[:] for seq in station_sequences]

            if move_type == "move":
                task = np.random.randint(0, self.num_tasks)
                old_s = int(new_station_assignment[task])
                new_s = int(np.random.randint(0, self.num_stations))
                if new_s == old_s:
                    continue

                if task in new_station_sequences[old_s]:
                    new_station_sequences[old_s].remove(task)
                new_station_sequences[new_s].append(task)
                new_station_assignment[task] = new_s

                if any(len(seq) == 0 for seq in new_station_sequences):
                    continue

                rand_keys = np.random.rand(self.num_tasks)
                new_station_sequences[old_s] = self.topo_sort_with_random_priority(new_station_sequences[old_s], rand_keys)
                new_station_sequences[new_s] = self.topo_sort_with_random_priority(new_station_sequences[new_s], rand_keys)

            else:
                s = int(np.random.randint(0, self.num_stations))
                if len(new_station_sequences[s]) < 2:
                    continue
                a, b = np.random.choice(new_station_sequences[s], size=2, replace=False)
                ia = new_station_sequences[s].index(a)
                ib = new_station_sequences[s].index(b)
                new_station_sequences[s][ia], new_station_sequences[s][ib] = new_station_sequences[s][ib], new_station_sequences[s][ia]
                rand_keys = np.random.rand(self.num_tasks)
                new_station_sequences[s] = self.topo_sort_with_random_priority(new_station_sequences[s], rand_keys)

            test_sol = (new_station_assignment, new_station_sequences)
            test_obj, _, _ = self.evaluate_objectives_with_penalty(test_sol)

            ct_ok = test_obj[0] <= best_obj[0] + self.eps_ct
            std_ok = test_obj[1] <= best_obj[1] + self.eps_std
            strictly_better = (test_obj[0] < best_obj[0] - self.eps_ct) or (test_obj[1] < best_obj[1] - self.eps_std)

            if ct_ok and std_ok and strictly_better:
                best_sol = test_sol
                best_obj = test_obj

        return best_sol, best_obj

    def optimize(self):
        num_qubits = self.num_tasks * (self.num_stations + 1)
        Q = np.zeros((self.pop_size, num_qubits, 2), dtype=float)
        Q[:, :, 0] = Q[:, :, 1] = 1 / np.sqrt(2)

        pareto_archive = []

        for t in range(self.max_iter):
            solutions = []
            original_objs = []
            penalized_objs = []
            X_obs = []

            progress = t / self.max_iter
            n_obs = self.n_obs_base + int(2 * progress) if progress < 1.0 else self.n_obs_base + 2

            for j in range(self.pop_size):
                sol, obs_bits, orig_obj = self.observe_individual(Q[j], n_obs, num_qubits)
                _, pen_obj, _ = self.evaluate_objectives_with_penalty(sol)

                if not self.has_empty_station(sol):
                    solutions.append(sol)
                    original_objs.append(orig_obj)
                    penalized_objs.append(pen_obj)
                    X_obs.append(obs_bits)

            X_obs = np.array(X_obs, dtype=int) if X_obs else np.empty((0, num_qubits), dtype=int)

            for sol, obj in zip(solutions, original_objs):
                pareto_archive = self.update_archive(pareto_archive, sol, obj)

            if self.use_local_improve and progress > 0.7 and pareto_archive:
                improved_archive = []
                for sol, obj in pareto_archive:
                    imp_sol, imp_obj = self.local_2obj_improvement(sol, obj, max_tries=60)
                    improved_archive.append((imp_sol, imp_obj))
                pareto_archive = improved_archive

            if pareto_archive:
                objs = np.array([obj for _, obj in pareto_archive], dtype=float)
                self.history["best_ct"].append(float(np.min(objs[:, 0])))
                self.history["best_load_std"].append(float(np.min(objs[:, 1])))
            else:
                self.history["best_ct"].append(np.inf)
                self.history["best_load_std"].append(np.inf)
            self.history["archive_size"].append(len(pareto_archive))

            guided_archive = [(sol, pen_obj) for sol, pen_obj in zip(solutions, penalized_objs)]
            Q = self.update_Q(Q, guided_archive, X_obs, t, self.max_iter)

            if t % 10 == 0:
                best_ct = self.history["best_ct"][-1]
                best_std = self.history["best_load_std"][-1]
                print(f"[IQEA] Iter {t:3d}/{self.max_iter:3d} | Archive: {len(pareto_archive):3d} | Valid: {len(guided_archive):3d} | Best CT: {best_ct:7.2f} | Best STD: {best_std:7.2f}")

        self.archive = pareto_archive
        return pareto_archive, self.history

    def plot_convergence(self, save_path: Path):
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(self.history["best_ct"], linewidth=2, marker='o', markersize=3)
        axes[0].set_xlabel("iteration")
        axes[0].set_ylabel("CT (s)")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history["best_load_std"], linewidth=2, marker='s', markersize=3)
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("LoadSTD (s)")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(self.history["archive_size"], linewidth=2, marker='d', markersize=3)
        axes[2].set_xlabel("iteration")
        axes[2].set_ylabel("Archive size")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


# ============================================================================
# Double-population GA
# ============================================================================
class ALBPDoubleGAEncoded:
    def __init__(
        self,
        tasks: Sequence[int],
        task_times: Sequence[int],
        precedence_matrix: np.ndarray,
        num_stations: int,
        pop_size: int = 30,
        max_gen: int = 100,
        exchange_interval: int = 10,
    ) -> None:
        self.tasks = list(tasks)
        self.task_times = list(task_times)
        self.precedence_matrix = precedence_matrix
        self.num_tasks = len(tasks)
        self.num_stations = num_stations
        self.total_time = sum(task_times)

        self.pop_size = pop_size
        self.max_gen = max_gen
        self.exchange_interval = exchange_interval

        self.population_explore: List[List[int]] = []
        self.population_exploit: List[List[int]] = []
        self.fitness_explore: List[float] = []
        self.fitness_exploit: List[float] = []

        self.best_chromosome: List[int] | None = None
        self.best_fitness = -float("inf")
        self.best_station_assignment: List[List[int]] | None = None
        self.best_population = ""
        self.best_actual_CT = float("inf")

        self.history_best_fitness: List[float] = []
        self.history_best_CT: List[float] = []
        self.history_load_std: List[float] = []
        self.history_avg_fitness_explore: List[float] = []
        self.history_avg_fitness_exploit: List[float] = []
        self.history_CT: List[float] = []
        self.history_LoadSTD: List[float] = []

        self.explore_pc1 = 0.9
        self.explore_pc2 = 0.75
        self.explore_pc3 = 0.6
        self.exploit_pc1 = 0.4
        self.exploit_pc2 = 0.3
        self.exploit_pc3 = 0.2

        self.explore_pm1 = 0.3
        self.explore_pm2 = 0.2
        self.explore_pm3 = 0.1
        self.exploit_pm1 = 0.1
        self.exploit_pm2 = 0.05
        self.exploit_pm3 = 0.01

        self.initialize_double_population()

    def _station_loads(self, station_assignment: Sequence[Sequence[int]]) -> List[float]:
        return [sum(self.task_times[t] for t in station) for station in station_assignment]

    def _calc_CT(self, station_assignment: Sequence[Sequence[int]]) -> float:
        loads = self._station_loads(station_assignment)
        return max(loads) if loads else 0.0

    def _calc_LoadSTD(self, station_assignment: Sequence[Sequence[int]]) -> float:
        loads = self._station_loads(station_assignment)
        if len(loads) <= 1:
            return 0.0
        return float(np.std(loads, ddof=1))

    def calculate_adaptive_pc(
        self, fitness: float, f_min: float, f_max: float, f_avg: float, is_explore: bool
    ) -> float:
        if f_avg == f_min:
            return self.explore_pc2 if is_explore else self.exploit_pc2
        if is_explore:
            pc1, pc2, pc3 = self.explore_pc1, self.explore_pc2, self.explore_pc3
        else:
            pc1, pc2, pc3 = self.exploit_pc1, self.exploit_pc2, self.exploit_pc3
        if fitness < f_avg:
            pc = (pc1 * (f_avg - fitness) + pc2 * (fitness - f_min)) / (f_avg - f_min)
        else:
            if f_max == f_avg:
                return pc2
            pc = (pc2 * (f_max - fitness) + pc3 * (fitness - f_avg)) / (f_max - f_avg)
        return max(0.1, min(0.9, pc))

    def calculate_adaptive_pm(
        self, fitness: float, f_min: float, f_max: float, f_avg: float, is_explore: bool
    ) -> float:
        if f_avg == f_min:
            return self.explore_pm2 if is_explore else self.exploit_pm2
        if is_explore:
            pm1, pm2, pm3 = self.explore_pm1, self.explore_pm2, self.explore_pm3
        else:
            pm1, pm2, pm3 = self.exploit_pm1, self.exploit_pm2, self.exploit_pm3
        if fitness < f_avg:
            pm = (pm1 * (f_avg - fitness) + pm2 * (fitness - f_min)) / (f_avg - f_min)
        else:
            if f_max == f_avg:
                return pm2
            pm = (pm2 * (f_max - fitness) + pm3 * (fitness - f_avg)) / (f_max - f_avg)
        return max(0.01, min(0.3, pm))

    def initialize_double_population(self) -> None:
        for _ in range(self.pop_size):
            self.population_explore.append(self.generate_random_feasible_chromosome())
        for _ in range(self.pop_size):
            self.population_exploit.append(self.generate_heuristic_feasible_chromosome())
        self.evaluate_population(self.population_explore, is_explore=True)
        self.evaluate_population(self.population_exploit, is_explore=False)

    def generate_random_feasible_chromosome(self) -> List[int]:
        chromosome: List[int] = []
        available_tasks = [i for i in range(self.num_tasks) if sum(self.precedence_matrix[:, i]) == 0]
        while available_tasks:
            task = random.choice(available_tasks)
            chromosome.append(task)
            available_tasks.remove(task)
            self._update_available_tasks(chromosome, available_tasks, task)
        return chromosome

    def generate_heuristic_feasible_chromosome(self) -> List[int]:
        chromosome: List[int] = []
        available_tasks = [i for i in range(self.num_tasks) if sum(self.precedence_matrix[:, i]) == 0]
        while available_tasks:
            available_tasks.sort(key=lambda x: self.task_times[x], reverse=True)
            task = available_tasks[0]
            chromosome.append(task)
            available_tasks.remove(task)
            self._update_available_tasks(chromosome, available_tasks, task)
        return chromosome

    def _update_available_tasks(
        self, chromosome: Sequence[int], available_tasks: List[int], current_task: int
    ) -> None:
        for j in range(self.num_tasks):
            if self.precedence_matrix[current_task, j] == 1:
                predecessors = [i for i in range(self.num_tasks) if self.precedence_matrix[i, j] == 1]
                if all(p in chromosome for p in predecessors) and j not in available_tasks:
                    available_tasks.append(j)

    def decode_chromosome(self, chromosome: Sequence[int]) -> Tuple[List[List[int]], float]:
        m = self.num_stations
        c0 = self.total_time / m
        stations: List[List[int]] = [[] for _ in range(m)]
        loads = [0.0] * m
        assigned = set()

        preds = {
            j: {i for i in range(self.num_tasks) if self.precedence_matrix[i, j] == 1}
            for j in range(self.num_tasks)
        }
        chrom_pos = {gene: idx for idx, gene in enumerate(chromosome)}

        while len(assigned) < self.num_tasks:
            progress = False
            for gene in chromosome:
                if gene in assigned:
                    continue
                if preds[gene].issubset(assigned):
                    order = sorted(range(m), key=lambda k: loads[k])
                    placed = False
                    for k in order:
                        if loads[k] + self.task_times[gene] <= c0:
                            stations[k].append(gene)
                            loads[k] += self.task_times[gene]
                            placed = True
                            break
                    if not placed:
                        k = min(range(m), key=lambda x: loads[x])
                        stations[k].append(gene)
                        loads[k] += self.task_times[gene]
                    assigned.add(gene)
                    progress = True
            if progress:
                continue

            feasible = [t for t in range(self.num_tasks) if t not in assigned and preds[t].issubset(assigned)]
            if not feasible:
                raise RuntimeError("No assignable task found. The precedence graph may contain a cycle.")

            gene = min(feasible, key=lambda g: chrom_pos.get(g, float("inf")))
            order = sorted(range(m), key=lambda k: loads[k])
            placed = False
            for k in order:
                if loads[k] + self.task_times[gene] <= c0:
                    stations[k].append(gene)
                    loads[k] += self.task_times[gene]
                    placed = True
                    break
            if not placed:
                k = min(range(m), key=lambda x: loads[x])
                stations[k].append(gene)
                loads[k] += self.task_times[gene]
            assigned.add(gene)

        return stations, max(loads) if loads else 0.0

    def evaluate_population(self, population: Sequence[Sequence[int]], is_explore: bool) -> None:
        fitness_list: List[float] = []
        total_ti = self.total_time
        m = self.num_stations
        for chromosome in population:
            station_assignment, max_station_time = self.decode_chromosome(chromosome)
            fitness = 0.0 if max_station_time == 0 else total_ti / (m * max_station_time)
            fitness_list.append(fitness)
            if (fitness > self.best_fitness) or (
                fitness == self.best_fitness and max_station_time < self.best_actual_CT
            ):
                self.best_fitness = fitness
                self.best_actual_CT = max_station_time
                self.best_chromosome = copy.deepcopy(list(chromosome))
                self.best_station_assignment = copy.deepcopy(station_assignment)
                self.best_population = "explore" if is_explore else "exploit"
        if is_explore:
            self.fitness_explore = fitness_list
        else:
            self.fitness_exploit = fitness_list

    def selection(self, population: Sequence[Sequence[int]], fitness: Sequence[float]) -> List[List[int]]:
        selected: List[List[int]] = []
        fitness_sum = max(sum(fitness), 1e-10)
        probabilities = [f / fitness_sum for f in fitness]
        for _ in range(len(population)):
            r = random.random()
            cumulative_prob = 0.0
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected.append(copy.deepcopy(list(population[i])))
                    break
        return selected

    def two_point_crossover(
        self, parent1: Sequence[int], parent2: Sequence[int], crossover_rate: float
    ) -> Tuple[List[int], List[int]]:
        if random.random() > crossover_rate:
            return copy.deepcopy(list(parent1)), copy.deepcopy(list(parent2))
        point1 = random.randint(0, self.num_tasks - 2)
        point2 = random.randint(point1 + 1, self.num_tasks - 1)
        child1: List[int | None] = [None] * self.num_tasks
        child2: List[int | None] = [None] * self.num_tasks
        child1[point1:point2] = parent2[point1:point2]
        child2[point1:point2] = parent1[point1:point2]
        self._fill_crossover_gaps(child1, parent1, point1, point2)
        self._fill_crossover_gaps(child2, parent2, point1, point2)
        return self.repair_chromosome(child1), self.repair_chromosome(child2)

    def _fill_crossover_gaps(
        self, child: List[int | None], parent: Sequence[int], point1: int, point2: int
    ) -> None:
        parent_genes = [gene for gene in parent if gene not in child[point1:point2]]
        child_gaps = [i for i in range(self.num_tasks) if i < point1 or i >= point2]
        for i, gene in zip(child_gaps, parent_genes):
            child[i] = gene

    def single_point_mutation(self, chromosome: Sequence[int], mutation_rate: float) -> List[int]:
        if random.random() > mutation_rate:
            return copy.deepcopy(list(chromosome))
        mutated = copy.deepcopy(list(chromosome))
        mutation_point = random.randint(1, self.num_tasks - 2)
        available_tasks = [i for i in range(self.num_tasks) if i not in mutated]
        if not available_tasks:
            pos1, pos2 = random.sample(range(self.num_tasks), 2)
            mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
        else:
            mutated[mutation_point] = random.choice(available_tasks)
        return self.repair_chromosome(mutated)

    def repair_chromosome(self, chromosome: Sequence[int | None]) -> List[int]:
        repaired: List[int] = []
        available_tasks = [i for i in range(self.num_tasks) if sum(self.precedence_matrix[:, i]) == 0]
        compact = [gene for gene in chromosome if gene is not None]
        if len(compact) < self.num_tasks:
            missing_tasks = set(range(self.num_tasks)) - set(compact)
            compact.extend(list(missing_tasks))
        while available_tasks:
            feasible_in_chrom = [task for task in compact if task in available_tasks]
            task = feasible_in_chrom[0] if feasible_in_chrom else random.choice(available_tasks)
            repaired.append(task)
            available_tasks.remove(task)
            self._update_available_tasks(repaired, available_tasks, task)
        return repaired

    def evolve_population(
        self, population: Sequence[Sequence[int]], fitness: Sequence[float], is_explore: bool
    ) -> List[List[int]]:
        f_min = min(fitness)
        f_max = max(fitness)
        f_avg = float(np.mean(fitness))
        selected = self.selection(population, fitness)
        random.shuffle(selected)
        pairs = [(selected[i], selected[i + 1]) for i in range(0, len(selected) - 1, 2)]

        new_pop: List[List[int]] = []
        for p1, p2 in pairs:
            f1 = self.calculate_fitness(p1)
            f2 = self.calculate_fitness(p2)
            pc_p1 = self.calculate_adaptive_pc(f1, f_min, f_max, f_avg, is_explore)
            pc_p2 = self.calculate_adaptive_pc(f2, f_min, f_max, f_avg, is_explore)
            c1, c2 = self.two_point_crossover(p1, p2, (pc_p1 + pc_p2) / 2.0)
            new_pop.extend([c1, c2])

        if len(selected) % 2 == 1:
            new_pop.append(selected[-1])

        for i in range(len(new_pop)):
            f = self.calculate_fitness(new_pop[i])
            pm = self.calculate_adaptive_pm(f, f_min, f_max, f_avg, is_explore)
            new_pop[i] = self.single_point_mutation(new_pop[i], pm)

        return new_pop[: self.pop_size]

    def calculate_fitness(self, chromosome: Sequence[int]) -> float:
        _, max_station_time = self.decode_chromosome(chromosome)
        return 0.0 if max_station_time == 0 else self.total_time / (self.num_stations * max_station_time)

    def exchange_best_individuals(self) -> None:
        explore_sorted_indices = np.argsort(self.fitness_explore)[::-1]
        exploit_sorted_indices = np.argsort(self.fitness_exploit)[::-1]
        best_explore_idx = int(explore_sorted_indices[0])
        best_exploit_idx = int(exploit_sorted_indices[0])
        best_explore = copy.deepcopy(self.population_explore[best_explore_idx])
        best_exploit = copy.deepcopy(self.population_exploit[best_exploit_idx])
        self.population_explore[best_explore_idx] = best_exploit
        self.population_exploit[best_exploit_idx] = best_explore
        self.evaluate_population(self.population_explore, is_explore=True)
        self.evaluate_population(self.population_exploit, is_explore=False)

    def optimize(self):
        for gen in range(self.max_gen):
            self.population_explore = self.evolve_population(
                self.population_explore, self.fitness_explore, is_explore=True
            )
            self.population_exploit = self.evolve_population(
                self.population_exploit, self.fitness_exploit, is_explore=False
            )
            self.evaluate_population(self.population_explore, is_explore=True)
            self.evaluate_population(self.population_exploit, is_explore=False)

            self.history_best_fitness.append(float(self.best_fitness))
            self.history_best_CT.append(float(self.best_actual_CT))
            self.history_avg_fitness_explore.append(float(np.mean(self.fitness_explore)))
            self.history_avg_fitness_exploit.append(float(np.mean(self.fitness_exploit)))

            if self.best_station_assignment is not None:
                ct = self._calc_CT(self.best_station_assignment)
                load_std = self._calc_LoadSTD(self.best_station_assignment)
            else:
                ct, load_std = 0.0, 0.0

            self.history_CT.append(float(ct))
            self.history_LoadSTD.append(float(load_std))
            self.history_load_std.append(float(load_std))

            if gen % self.exchange_interval == 0 and gen > 0:
                self.exchange_best_individuals()

            if gen % 10 == 0:
                print(f"[DoubleGA] Gen {gen:3d} | Best fitness: {self.best_fitness:.4f} | CT: {ct:.2f} | LoadSTD: {load_std:.2f}")

        return self.best_chromosome, float(self.best_fitness), self.best_station_assignment

    def plot_convergence(self, save_path: Path) -> None:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        generations = range(len(self.history_best_fitness))
        ax1.plot(generations, self.history_best_fitness, "b-", label="Global best fitness")
        ax1.plot(generations, self.history_avg_fitness_explore, "m--", label="Explore avg fitness")
        ax1.plot(generations, self.history_avg_fitness_exploit, "y--", label="Exploit avg fitness")
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("Fitness")
        ax1.legend(loc="upper right")
        ax1.grid(True)
        ax2.plot(generations, self.history_CT, "g-", label="Optimal CT")
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("CT (s)")
        ax2.legend()
        ax2.grid(True)
        ax3.plot(generations, self.history_LoadSTD, "r-", label="LoadSTD")
        ax3.set_xlabel("iteration")
        ax3.set_ylabel("LoadSTD (s)")
        ax3.legend()
        ax3.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


# ============================================================================
# IPSO
# ============================================================================
class ALBP_PSO:
    def __init__(self, tasks, task_times, precedence_matrix, num_stations,
                 num_particles=30, max_iter=100, w_max=0.9, w_min=0.4, c1=1.5, c2=1.5, v_max=5.0,
                 obli_prob=0.3, rob_obli_prob=0.1):
        self.tasks = tasks
        self.task_times = task_times
        self.precedence_matrix = precedence_matrix
        self.num_tasks = len(tasks)
        self.num_stations = num_stations
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.obli_prob = obli_prob
        self.rob_obli_prob = rob_obli_prob

        self.total_time = sum(self.task_times)
        self.max_task_time = max(self.task_times)

        self.particles = []
        self.velocities = []
        self.pbest_positions = []
        self.pbest_values = []
        self.gbest_position = None
        self.gbest_value = float('inf')
        self.gbest_station_assignment = None
        self.gbest_cycle_time = float('inf')
        self.gbest_std_dev = float('inf')

        self.dim_min = np.zeros(self.num_tasks) - 0.5
        self.dim_max = np.ones(self.num_tasks) * 0.5

        self.best_fitness_history = []
        self.best_cycle_time_history = []
        self.best_std_dev_history = []
        self.current_cycle_time_history = []

        self.initialize_particles()

    def initialize_particles(self):
        for _ in range(self.num_particles):
            priority_values = np.random.uniform(-0.5, 0.5, self.num_tasks)
            velocity = np.zeros(self.num_tasks)
            self.particles.append(priority_values)
            self.velocities.append(velocity)
            self.pbest_positions.append(priority_values.copy())
            sequence = self.decode_priority_to_sequence(priority_values)
            fitness, station_assignment, cycle_time, std_dev = self.calculate_fitness(sequence)
            self.pbest_values.append(fitness)
            if fitness < self.gbest_value:
                self.gbest_value = fitness
                self.gbest_position = priority_values.copy()
                self.gbest_station_assignment = station_assignment
                self.gbest_cycle_time = cycle_time
                self.gbest_std_dev = std_dev

    def decode_priority_to_sequence(self, priority_values):
        sequence = []
        remaining_precedence = self.precedence_matrix.copy()
        remaining_tasks = set(range(self.num_tasks))

        while remaining_tasks:
            available_tasks = []
            for task in remaining_tasks:
                has_predecessors = False
                for j in range(self.num_tasks):
                    if remaining_precedence[j, task] == 1 and j in remaining_tasks:
                        has_predecessors = True
                        break
                if not has_predecessors:
                    available_tasks.append(task)

            if not available_tasks:
                task = max(remaining_tasks, key=lambda x: priority_values[x])
            else:
                task = max(available_tasks, key=lambda x: priority_values[x])

            sequence.append(task)
            remaining_tasks.remove(task)
            for j in range(self.num_tasks):
                remaining_precedence[task, j] = 0
        return sequence

    def assign_to_stations(self, sequence):
        c = max(self.total_time / self.num_stations, 1.5 * self.max_task_time)
        iteration = 0
        while iteration < self.max_iter:
            iteration += 1
            station_assignment = [[] for _ in range(self.num_stations)]
            station_times = np.zeros(self.num_stations)
            task_index = 0
            for k in range(self.num_stations - 1):
                while task_index < len(sequence):
                    task = sequence[task_index]
                    task_time = self.task_times[task]
                    if station_times[k] + task_time <= c:
                        station_assignment[k].append(task)
                        station_times[k] += task_time
                        task_index += 1
                    else:
                        break

            station_assignment[-1] = sequence[task_index:]
            station_times[-1] = sum(self.task_times[task] for task in station_assignment[-1])

            t = station_times[-1]
            if t <= c:
                actual_cycle_time = max(station_times)
                return actual_cycle_time, station_assignment

            potential_times = []
            for k in range(self.num_stations - 1):
                if task_index < len(sequence):
                    next_task = sequence[task_index]
                    potential_times.append(station_times[k] + self.task_times[next_task])
                else:
                    potential_times.append(float('inf'))

            c1 = min(potential_times) if potential_times else float('inf')
            c = t if t <= c1 else c1
            actual_cycle_time = max(station_times)
            return actual_cycle_time, station_assignment

    def calculate_fitness(self, sequence):
        cycle_time, station_assignment = self.assign_to_stations(sequence)
        station_times = [sum(self.task_times[task] for task in station) for station in station_assignment]
        actual_cycle_time = max(station_times)
        if len(station_times) > 1:
            deviations = [t - actual_cycle_time for t in station_times]
            variance = sum(d ** 2 for d in deviations) / (len(station_times) - 1)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.0
        fitness = 0.5 * actual_cycle_time + 0.5 * std_dev
        return fitness, station_assignment, actual_cycle_time, std_dev

    def update_velocity(self, i, w):
        r1 = random.random()
        r2 = random.random()
        cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.particles[i])
        social = self.c2 * r2 * (self.gbest_position - self.particles[i])
        new_velocity = w * self.velocities[i] + cognitive + social
        self.velocities[i] = np.clip(new_velocity, -self.v_max, self.v_max)

    def update_position(self, i):
        new_position = np.clip(self.particles[i] + self.velocities[i], -0.5, 0.5)
        self.particles[i] = new_position

        sequence = self.decode_priority_to_sequence(new_position)
        new_fitness, new_station_assignment, new_cycle_time, new_std_dev = self.calculate_fitness(sequence)

        if new_fitness < self.pbest_values[i]:
            self.pbest_values[i] = new_fitness
            self.pbest_positions[i] = new_position.copy()

            if new_fitness < self.gbest_value:
                self.gbest_value = new_fitness
                self.gbest_position = new_position.copy()
                self.gbest_station_assignment = new_station_assignment
                self.gbest_cycle_time = new_cycle_time
                self.gbest_std_dev = new_std_dev

    def optimize(self):
        for iteration in range(self.max_iter):
            w = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iter

            for i in range(self.num_particles):
                self.update_velocity(i, w)
                self.update_position(i)

            self.best_fitness_history.append(float(self.gbest_value))
            self.best_cycle_time_history.append(float(self.gbest_cycle_time))
            self.best_std_dev_history.append(float(self.gbest_std_dev))

            seq = self.decode_priority_to_sequence(self.gbest_position)
            _, _, cur_cycle_time, _ = self.calculate_fitness(seq)
            self.current_cycle_time_history.append(float(cur_cycle_time))

            if iteration % 10 == 0:
                print(f"[IPSO] Iter {iteration:3d} | Fitness: {self.gbest_value:.4f} | CT: {self.gbest_cycle_time:.2f} | LoadSTD: {self.gbest_std_dev:.2f}")

        best_sequence = self.decode_priority_to_sequence(self.gbest_position)
        _, _, final_cycle_time, final_std_dev = self.calculate_fitness(best_sequence)
        return best_sequence, final_cycle_time, final_std_dev, self.gbest_station_assignment

    def plot_convergence(self, save_path: Path):
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 10))

        plt.subplot(3, 1, 1)
        plt.plot(self.best_fitness_history, 'b-', linewidth=2, label='Global Optimal Fitness')
        plt.xlabel('iteration')
        plt.ylabel('Fitness')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(self.current_cycle_time_history, 'g-', linewidth=2, label='Optimal Takt Time')
        plt.xlabel('iteration')
        plt.ylabel('CT (s)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.best_std_dev_history, 'r-', linewidth=2, label='Standard Deviation of Workstation Load')
        plt.xlabel('iteration')
        plt.ylabel('LoadSTD (s)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()


# ============================================================================
# Saving helpers for each algorithm
# ============================================================================
def save_iqea_outputs(model: TwoObjectiveIQEA, out_dir: Path, instance_name: str, seed: int, precedence_relations):
    out_dir.mkdir(parents=True, exist_ok=True)

    reps = model.select_representative_solutions(model.archive)
    summary = {
        "algorithm": "iqea",
        "instance": instance_name,
        "seed": seed,
        "archive_size": len(model.archive),
        "best_ct_over_history": model.history["best_ct"][-1] if model.history["best_ct"] else None,
        "best_load_std_over_history": model.history["best_load_std"][-1] if model.history["best_load_std"] else None,
        "representative_solution_tags": [tag for tag, _ in reps],
        "precedence_relations_0based": precedence_relations,
    }
    (out_dir / "result_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "convergence_history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "best_CT", "best_LoadSTD", "archive_size"])
        for i in range(len(model.history["best_ct"])):
            writer.writerow([
                i + 1,
                model.history["best_ct"][i],
                model.history["best_load_std"][i],
                model.history["archive_size"][i],
            ])

    with (out_dir / "pareto_archive.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["solution_id", "CT", "LoadSTD", "station_loads", "station_sequences_1based"])
        for idx, (sol, obj) in enumerate(model.archive, start=1):
            station_assignment, station_sequences = sol
            station_times = model.compute_station_times(sol)
            seq_1based = [[t + 1 for t in seq] for seq in station_sequences]
            writer.writerow([
                idx,
                obj[0],
                obj[1],
                " ".join([f"{x:.6f}" for x in station_times]),
                json.dumps(seq_1based, ensure_ascii=False),
            ])

    with (out_dir / "representative_solutions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "CT", "LoadSTD", "station_loads", "station_sequences_1based"])
        for tag, (sol, obj) in reps:
            station_assignment, station_sequences = sol
            station_times = model.compute_station_times(sol)
            seq_1based = [[t + 1 for t in seq] for seq in station_sequences]
            writer.writerow([
                tag,
                obj[0],
                obj[1],
                " ".join([f"{x:.6f}" for x in station_times]),
                json.dumps(seq_1based, ensure_ascii=False),
            ])

        # Save first representative solution as separate station assignment csv for convenience
        if reps:
            first_sol = reps[0][1][0]
            save_station_assignment_csv(out_dir / "representative_station_assignment.csv", first_sol[1], model.task_times)


def save_doublega_outputs(model, out_dir: Path, instance_name: str, seed: int, precedence_relations):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "algorithm": "doublega",
        "instance": instance_name,
        "seed": seed,
        "best_fitness": model.best_fitness,
        "best_CT": model.best_actual_CT,
        "best_LoadSTD": model._calc_LoadSTD(model.best_station_assignment) if model.best_station_assignment is not None else None,
        "best_population": model.best_population,
        "best_chromosome_1based": [x + 1 for x in model.best_chromosome] if model.best_chromosome is not None else None,
        "station_assignment_1based": [[x + 1 for x in st] for st in model.best_station_assignment] if model.best_station_assignment is not None else None,
        "precedence_relations_0based": precedence_relations,
    }
    (out_dir / "result_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "convergence_history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "best_CT", "best_LoadSTD", "avg_fitness_explore", "avg_fitness_exploit"])
        for i in range(len(model.history_best_fitness)):
            writer.writerow([
                i + 1,
                model.history_best_fitness[i],
                model.history_CT[i],
                model.history_LoadSTD[i],
                model.history_avg_fitness_explore[i],
                model.history_avg_fitness_exploit[i],
            ])

    if model.best_station_assignment is not None:
        save_station_assignment_csv(out_dir / "best_station_assignment.csv", model.best_station_assignment, model.task_times)


def save_ipso_outputs(model, out_dir: Path, instance_name: str, seed: int, best_sequence, best_ct, best_std, precedence_relations):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "algorithm": "ipso",
        "instance": instance_name,
        "seed": seed,
        "best_fitness": model.gbest_value,
        "best_CT": best_ct,
        "best_LoadSTD": best_std,
        "best_sequence_1based": [x + 1 for x in best_sequence],
        "station_assignment_1based": [[x + 1 for x in st] for st in model.gbest_station_assignment] if model.gbest_station_assignment is not None else None,
        "precedence_relations_0based": precedence_relations,
    }
    (out_dir / "result_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "convergence_history.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "best_fitness", "best_CT", "best_LoadSTD"])
        for i in range(len(model.best_fitness_history)):
            writer.writerow([
                i + 1,
                model.best_fitness_history[i],
                model.current_cycle_time_history[i],
                model.best_std_dev_history[i],
            ])

    if model.gbest_station_assignment is not None:
        save_station_assignment_csv(out_dir / "best_station_assignment.csv", model.gbest_station_assignment, model.task_times)


# ============================================================================
# CLI
# ============================================================================
@dataclass
class RunConfig:
    algorithm: str
    instance: str
    seed: int
    out_dir: str
    save_plot: bool
    pop_size: int
    max_gen: int
    exchange_interval: int
    iqea_max_iter: int
    iqea_n_obs_base: int
    iqea_use_local_improve: bool
    num_particles: int
    max_iter: int
    w_max: float
    w_min: float
    c1: float
    c2: float
    v_max: float


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Combined clean reproducibility script for IQEA, DoubleGA, and IPSO.")
    parser.add_argument("--algorithm", choices=["iqea", "doublega", "ipso"], default="iqea")
    parser.add_argument("--instance", choices=["kilbridge45", "weld24"], default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--out-dir", type=str, default="combined_results")
    parser.add_argument("--no-plot", action="store_true")

    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--max-gen", type=int, default=100)
    parser.add_argument("--exchange-interval", type=int, default=10)

    parser.add_argument("--iqea-max-iter", type=int, default=100)
    parser.add_argument("--iqea-n-obs-base", type=int, default=3)
    parser.add_argument("--iqea-no-local-improve", action="store_true")

    parser.add_argument("--num-particles", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--w-max", type=float, default=0.9)
    parser.add_argument("--w-min", type=float, default=0.4)
    parser.add_argument("--c1", type=float, default=2.0)
    parser.add_argument("--c2", type=float, default=2.0)
    parser.add_argument("--v-max", type=float, default=0.05)

    args = parser.parse_args()

    instance = args.instance
    if instance is None:
        if args.algorithm in {"iqea", "doublega"}:
            instance = "kilbridge45"
        else:
            instance = "weld24"

    return RunConfig(
        algorithm=args.algorithm,
        instance=instance,
        seed=args.seed,
        out_dir=args.out_dir,
        save_plot=not args.no_plot,
        pop_size=args.pop_size,
        max_gen=args.max_gen,
        exchange_interval=args.exchange_interval,
        iqea_max_iter=args.iqea_max_iter,
        iqea_n_obs_base=args.iqea_n_obs_base,
        iqea_use_local_improve=not args.iqea_no_local_improve,
        num_particles=args.num_particles,
        max_iter=args.max_iter,
        w_max=args.w_max,
        w_min=args.w_min,
        c1=args.c1,
        c2=args.c2,
        v_max=args.v_max,
    )


def main():
    config = parse_args()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    np.random.seed(config.seed)

    tasks, task_times, precedence_matrix, num_stations, precedence_relations = load_instance(config.instance)

    if config.algorithm == "iqea":
        model = TwoObjectiveIQEA(
            tasks=tasks,
            task_times=task_times,
            precedence_matrix=precedence_matrix,
            num_stations=num_stations,
            pop_size=config.pop_size,
            max_iter=config.iqea_max_iter,
            n_obs_base=config.iqea_n_obs_base,
            use_local_improve=config.iqea_use_local_improve,
        )
        archive, history = model.optimize()
        algo_dir = out_dir / "iqea"
        save_iqea_outputs(model, algo_dir, config.instance, config.seed, precedence_relations)
        if config.save_plot:
            model.plot_convergence(algo_dir / "convergence_curve_iqea.png")

        print("=" * 72)
        print("IQEA finished.")
        print(f"Output directory : {algo_dir.resolve()}")
        print(f"Archive size     : {len(archive)}")
        if history["best_ct"]:
            print(f"Best CT          : {history['best_ct'][-1]:.6f}")
            print(f"Best LoadSTD     : {history['best_load_std'][-1]:.6f}")
        print("=" * 72)

    elif config.algorithm == "doublega":
        model = ALBPDoubleGAEncoded(
            tasks=tasks,
            task_times=task_times,
            precedence_matrix=precedence_matrix,
            num_stations=num_stations,
            pop_size=config.pop_size,
            max_gen=config.max_gen,
            exchange_interval=config.exchange_interval,
        )
        model.optimize()
        algo_dir = out_dir / "doublega"
        save_doublega_outputs(model, algo_dir, config.instance, config.seed, precedence_relations)
        if config.save_plot:
            model.plot_convergence(algo_dir / "convergence_curve_doublega.png")
        print("=" * 72)
        print("DoubleGA finished.")
        print(f"Output directory : {algo_dir.resolve()}")
        print(f"Best fitness     : {model.best_fitness:.6f}")
        print(f"Best CT          : {model.best_actual_CT:.6f}")
        if model.best_station_assignment is not None:
            print(f"Best LoadSTD     : {model._calc_LoadSTD(model.best_station_assignment):.6f}")
        print("=" * 72)

    elif config.algorithm == "ipso":
        model = ALBP_PSO(
            tasks=tasks,
            task_times=task_times,
            precedence_matrix=precedence_matrix,
            num_stations=num_stations,
            num_particles=config.num_particles,
            max_iter=config.max_iter,
            w_max=config.w_max,
            w_min=config.w_min,
            c1=config.c1,
            c2=config.c2,
            v_max=config.v_max,
            obli_prob=0.2,
            rob_obli_prob=0.1,
        )
        best_seq, best_ct, best_std, _ = model.optimize()
        algo_dir = out_dir / "ipso"
        save_ipso_outputs(model, algo_dir, config.instance, config.seed, best_seq, best_ct, best_std, precedence_relations)
        if config.save_plot:
            model.plot_convergence(algo_dir / "convergence_curve_ipso.png")
        print("=" * 72)
        print("IPSO finished.")
        print(f"Output directory : {algo_dir.resolve()}")
        print(f"Best fitness     : {model.gbest_value:.6f}")
        print(f"Best CT          : {best_ct:.6f}")
        print(f"Best LoadSTD     : {best_std:.6f}")
        print("=" * 72)


if __name__ == "__main__":
    main()
