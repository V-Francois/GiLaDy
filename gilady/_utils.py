from __future__ import annotations

import dataclasses

import numpy as np
import pymser

PRECISION_THRESHOLD = 1e-4


@dataclasses.dataclass
class LambdaVector:
    lambdas: list[float]

    def __post_init__(self) -> None:
        if abs(np.sum(self.lambdas) - 1) > PRECISION_THRESHOLD:
            raise ValueError("lambdas do not sum to one!")


def lambda_vectors_from_single_values(lambdas: list[float], number_of_states: int) -> list[LambdaVector]:
    lambda_vector = [[x] for x in lambdas]
    for _ in range(number_of_states - 1):
        lambda_vector = [[*x, y] for y in lambdas for x in lambda_vector]
    # Only keep the ones for which the sum of individual lambas is one
    return [LambdaVector(lambdas=x) for x in lambda_vector if abs(np.sum(x) - 1) < PRECISION_THRESHOLD]


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
