# from __future__ import annotations
import dataclasses
import tempfile
from collections.abc import Callable

import _utils
import openmm
from openmm import app, unit


@dataclasses.dataclass
class GiLaDy:
    lambda_updator: Callable[[openmm.Context, float], None]
    current_lambda: float = 1.0
    simulation: app.Simulation | None = None

    def _update_lambda(self, new_lambda: float) -> None:
        # TODO: do it slowly to avoid insane clashes when moving from no LJ to LJ
        self.lambda_updator(self.simulation.context, new_lambda)
        self.simulation.minimizeEnergy()

    def find_characteristic_time(
        self,
        simulation: app.Simulation,
        max_time_to_consider: unit.Quantity = 20 * unit.picosecond,
    ) -> dict:
        self.simulation = simulation

        time_between_prints = 0.1 * unit.picosecond
        steps_between_prints = int(time_between_prints / simulation.integrator.getStepSize())
        steps_between_lambda_changes = int(max_time_to_consider / simulation.integrator.getStepSize())

        # Equilibrate
        self._update_lambda(1.0)
        simulation.minimizeEnergy()
        simulation.step(steps_between_lambda_changes)

        n_round_trips = 10
        times_dict = {"0_to_1": [], "1_to_0": []}
        tmp_file = tempfile.NamedTemporaryFile()
        for _ in range(n_round_trips):
            for target_lambda in [0.0, 1.0]:
                # Go from lambda = 1 to lambda = 0
                self._update_lambda(target_lambda)
                simulation.minimizeEnergy()
                # self._update_lambda(1.0)
                reporter = app.statedatareporter.StateDataReporter(
                    tmp_file.name,
                    steps_between_prints,
                    potentialEnergy=True,
                )
                simulation.reporters.append(reporter)
                simulation.step(steps_between_lambda_changes)
                simulation.reporters.pop()

                with open(tmp_file.name) as energy_file:
                    potential_energy = [float(x.strip()) for x in energy_file.readlines()[1:]]
                nb_steps_to_reach_eq = _utils.find_number_of_steps_to_reach_equilibrium(potential_energy)
                if target_lambda == 0:
                    times_dict["1_to_0"].append(nb_steps_to_reach_eq * time_between_prints)
                else:
                    times_dict["0_to_1"].append(nb_steps_to_reach_eq * time_between_prints)
        return times_dict
