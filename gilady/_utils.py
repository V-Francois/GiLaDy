from __future__ import annotations

import numpy as np
import pymser


def lambda_vectors_from_single_values(lambdas: list[float], number_of_states: int) -> list[list[float]]:
    lambda_vector = [[x] for x in lambdas]
    for _ in range(number_of_states - 1):
        lambda_vector = [[*x, y] for y in lambdas for x in lambda_vector]
    # Only keep the ones for which the sum of individual lambas is one
    return [x for x in lambda_vector if abs(np.sum(x) - 1) < 1e-4]


def find_number_of_steps_to_reach_equilibrium(timeseries: list[float]) -> int:
    results = pymser.equilibrate(
        timeseries,
        LLM=True,
        batch_size=1,
        ADF_test=True,
        uncertainty="uSD",
        print_results=False,
    )

    return results["t0"]
