# from __future__ import annotations
import dataclasses
import tempfile
from collections.abc import Callable

import _utils
import numpy as np
import openmm
from openmm import app, unit
from pymbar import mbar


@dataclasses.dataclass
class GiLaDy:
    simulation: app.Simulation
    lambda_updator: Callable[[openmm.Context, float], None]
    current_lambda: float = 1.0
    number_swap_before_bias_update: int = 10
    number_cycle_of_bias_update: int = 4
    lambdas: list[float] | None = None

    def _update_lambda(self, new_lambda: float) -> None:
        # Large changes in lambda, especially from a fully non interacting system
        # to a interacting one, could create big clashes and result in NaNs
        # So we do a series of minimizations for some intermediary values of lambda
        for intermediary_lambda in np.linspace(
            self.current_lambda,
            new_lambda,
            num=int(abs((self.current_lambda - new_lambda) / 0.05)),
        ):
            self.lambda_updator(self.simulation.context, intermediary_lambda)
            self.simulation.minimizeEnergy()
        self.lambda_updator(self.simulation.context, new_lambda)
        self.simulation.minimizeEnergy()
        self.current_lambda = new_lambda

    def find_characteristic_time(
        self,
        max_time_to_consider: unit.Quantity = 20 * unit.picosecond,
    ) -> dict:
        time_between_prints = 0.1 * unit.picosecond
        steps_between_prints = int(time_between_prints / self.simulation.integrator.getStepSize())
        steps_between_lambda_changes = int(max_time_to_consider / self.simulation.integrator.getStepSize())

        # Equilibrate
        self._update_lambda(1.0)
        self.simulation.minimizeEnergy()
        self.simulation.step(steps_between_lambda_changes)

        n_round_trips = 10
        times_dict = {"0_to_1": [], "1_to_0": []}
        tmp_file = tempfile.NamedTemporaryFile()
        for _ in range(n_round_trips):
            for target_lambda in [0.0, 1.0]:
                # Go from lambda = 1 to lambda = 0
                self._update_lambda(target_lambda)
                self.simulation.minimizeEnergy()
                # self._update_lambda(1.0)
                reporter = app.statedatareporter.StateDataReporter(
                    tmp_file.name,
                    steps_between_prints,
                    potentialEnergy=True,
                )
                self.simulation.reporters.append(reporter)
                self.simulation.step(steps_between_lambda_changes)
                self.simulation.reporters.pop()

                with open(tmp_file.name) as energy_file:
                    potential_energy = [float(x.strip()) for x in energy_file.readlines()[1:]]
                nb_steps_to_reach_eq = _utils.find_number_of_steps_to_reach_equilibrium(potential_energy)
                if target_lambda == 0:
                    times_dict["1_to_0"].append(nb_steps_to_reach_eq * time_between_prints)
                else:
                    times_dict["0_to_1"].append(nb_steps_to_reach_eq * time_between_prints)
        return times_dict

    def _select_new_lambda(self) -> float:
        return np.random.choice(self.lambdas)

    def _update_biases(self) -> None:
        pass

    def _compute_energies(self) -> np.ndarray:
        energies = []
        for lambda_value in self.lambdas:
            self.lambda_updator(self.simulation.context, lambda_value)
            energies.append(self.simulation.context.getState(getEnergy=True).getPotentialEnergy())
        return np.array([e.value_in_unit(unit.kilocalorie_per_mole) for e in energies])

    def _run_mbar(self, energies: np.ndarray, lambda_sampled: np.ndarray) -> np.ndarray:
        kbt = (self.simulation.integrator.getTemperature() * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
            unit.kilocalorie_per_mole
        )

        # Order energies
        u_nk = np.zeros(energies.shape)
        n_k = np.zeros(len(self.lambdas))
        counter = 0
        for lambda_index, lambda_value in enumerate(self.lambdas):
            indices_where_lambda_was_sampled = np.where([lbd == lambda_value for lbd in lambda_sampled])[0]
            n_k[lambda_index] = len(indices_where_lambda_was_sampled)
            for index in indices_where_lambda_was_sampled:
                u_nk[counter, :] = energies[index, :] / kbt
                counter += 1
        mbar_object = mbar.MBAR(u_nk.T, n_k)
        return mbar_object.compute_free_energy_differences()["Delta_f"][0] * kbt

    def run(
        self,
        lambdas: list[float],
        time_between_lambda_change: unit.Quantity = 10 * unit.picosecond,
    ) -> np.ndarray:
        self.lambdas = lambdas

        total_number_of_energy_evaluations = self.number_swap_before_bias_update * self.number_cycle_of_bias_update
        energies = np.zeros((total_number_of_energy_evaluations, len(lambdas)))

        # Start from a random lambda state
        self.current_lambda = np.random.choice(lambdas)
        self._update_lambda(self.current_lambda)

        number_md_steps_between_lambda_change = int(time_between_lambda_change / self.simulation.integrator.getStepSize())
        lambda_sampled = []
        for bias_cycle in range(self.number_cycle_of_bias_update):
            for swap_cycle in range(self.number_swap_before_bias_update):
                new_lambda = self._select_new_lambda()
                self._update_lambda(new_lambda)
                lambda_sampled.append(new_lambda)
                self.simulation.step(number_md_steps_between_lambda_change)

                energy_index = swap_cycle + bias_cycle * self.number_swap_before_bias_update
                print(energy_index)
                energies[energy_index, :] = self._compute_energies()
            self._update_biases()

        return self._run_mbar(energies, lambda_sampled)
