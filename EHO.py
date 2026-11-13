import time
import numpy as np


# Elk Herd Optimizer (EHO)
def EHO(X, objective_function, VRmin, VRmax, max_iter):

    num_elks = X.shape[0]
    dim = X.shape[1]

    lb = VRmin[0, :]
    ub = VRmax[0, :]

    # Initial Fitness
    fitness = np.array([objective_function(ind) for ind in X])

    best_idx = np.argmin(fitness)
    Leader = X[best_idx].copy()
    Leader_score = fitness[best_idx]

    Convergence_curve = np.zeros((max_iter, 1))

    start_time = time.time()

    for t in range(max_iter):

        print(f"Iteration {t + 1}/{max_iter}")

        alpha = 2 * (1 - t / max_iter)
        beta = np.random.rand()

        mean_position = np.mean(X, axis=0)

        # Herd Movement Phase
        for i in range(num_elks):

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            X_new = (
                X[i]
                + alpha * r1 * (Leader - X[i])
                + beta * r2 * (mean_position - X[i])
            )

            X_new = np.clip(X_new, lb, ub)

            new_fit = objective_function(X_new)

            if new_fit < fitness[i]:
                X[i] = X_new.copy()
                fitness[i] = new_fit

            if new_fit < Leader_score:
                Leader_score = new_fit
                Leader = X_new.copy()

        # Competition Phase
        for i in range(num_elks):

            rival_idx = np.random.randint(num_elks)

            while rival_idx == i:
                rival_idx = np.random.randint(num_elks)

            if fitness[rival_idx] < fitness[i]:

                X_new = (
                    X[i]
                    + np.random.rand(dim)
                    * (X[rival_idx] - X[i])
                )

            else:

                X_new = (
                    X[i]
                    + np.random.rand(dim)
                    * (Leader - X[i])
                )

            X_new = np.clip(X_new, lb, ub)

            new_fit = objective_function(X_new)

            if new_fit < fitness[i]:
                X[i] = X_new.copy()
                fitness[i] = new_fit

            if new_fit < Leader_score:
                Leader_score = new_fit
                Leader = X_new.copy()

        # Exploration Phase
        exploration_rate = 0.1

        for i in range(num_elks):

            if np.random.rand() < exploration_rate:

                X_new = lb + np.random.rand(dim) * (ub - lb)

                new_fit = objective_function(X_new)

                if new_fit < fitness[i]:
                    X[i] = X_new.copy()
                    fitness[i] = new_fit

                if new_fit < Leader_score:
                    Leader_score = new_fit
                    Leader = X_new.copy()

        Convergence_curve[t] = Leader_score

    total_time = time.time() - start_time

    return Leader_score, Convergence_curve, Leader, total_time