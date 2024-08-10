from __future__ import annotations
import pymser


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
