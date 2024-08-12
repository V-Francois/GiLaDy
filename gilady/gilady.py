import dataclasses
import tempfile
from collections.abc import Callable

import _utils
import numpy as np
import openmm
from openmm import app, unit
from pymbar import mbar

# TODO: return some transition statisics: do you jump mostly to adjactent lambdas, or can you have large jumps?


@dataclasses.dataclass
class GiLaDy:
    # To setup at init
    simulation: app.Simulation
    lambda_updator: Callable[[openmm.Context, float], None]

    # Not to be changed by the user
    # TODO: make private
    current_lambda: float = 1.0
    lambdas: list[float] = dataclasses.field(default_factory=list)
    lambdas_sampled: list[float] = dataclasses.field(default_factory=list)
    lambdas_counts: np.ndarray = dataclasses.field(default_factory=lambda: np.array([], dtype=int))
    current_biases: np.ndarray = dataclasses.field(default_factory=lambda: np.array([], dtype=float))
    free_energy_convergence: list[np.ndarray] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        """Setups constants."""
        self.kbt = (self.simulation.integrator.getTemperature() * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
            unit.kilocalorie_per_mole
        )

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
        """Find characteristic time of the lambda change."""
        time_between_prints = 0.1 * unit.picosecond
        steps_between_prints = int(time_between_prints / self.simulation.integrator.getStepSize())
        steps_between_lambda_changes = int(max_time_to_consider / self.simulation.integrator.getStepSize())

        # Equilibrate
        self._update_lambda(1.0)
        self.simulation.minimizeEnergy()
        self.simulation.step(steps_between_lambda_changes)

        n_round_trips = 10
        times_dict: dict[str, list[float]] = {"0_to_1": [], "1_to_0": []}
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

    def _select_new_lambda(self, energy_this_step: np.ndarray) -> float:
        # Eq.4, with bigger bias at first run
        penalty = 10.0 if all(self.current_biases == 0) else 1.0
        bias_based_on_frequency = penalty * 2 ** (self.lambdas_counts - min(self.lambdas_counts))
        energy_plus_bias = energy_this_step + self.current_biases + bias_based_on_frequency

        probas = np.exp(-(energy_plus_bias - min(energy_plus_bias)) / self.kbt)
        probas /= np.sum(probas)

        new_lambda = np.random.choice(self.lambdas, p=probas)

        return new_lambda

    def _update_biases(self, energies: np.ndarray) -> None:
        current_free_energies = self._get_free_energy_from_mbar(energies)
        self.free_energy_convergence.append(current_free_energies)
        self.current_biases = -current_free_energies

    def _compute_energies(self) -> np.ndarray:
        energies = []
        for lambda_value in self.lambdas:
            self.lambda_updator(self.simulation.context, lambda_value)
            energies.append(self.simulation.context.getState(getEnergy=True).getPotentialEnergy())
        return np.array([e.value_in_unit(unit.kilocalorie_per_mole) for e in energies])

    def _get_free_energy_from_mbar(self, energies: np.ndarray) -> np.ndarray:
        mbar_object = self._run_mbar(energies)

        return np.array(mbar_object.compute_free_energy_differences()["Delta_f"][0] * self.kbt)

    def _run_mbar(self, energies: np.ndarray) -> mbar.MBAR:
        # Order energies
        u_nk = np.zeros(energies.shape)
        n_k = np.zeros(len(self.lambdas))
        counter = 0
        for lambda_index, lambda_value in enumerate(self.lambdas):
            indices_where_lambda_was_sampled = np.where([lbd == lambda_value for lbd in self.lambdas_sampled])[0]
            n_k[lambda_index] = len(indices_where_lambda_was_sampled)
            for index in indices_where_lambda_was_sampled:
                u_nk[counter, :] = energies[index, :] / self.kbt
                counter += 1
        return mbar.MBAR(u_nk.T, n_k)

    def run(
        self,
        lambdas: list[float],
        number_swap_before_bias_update: int = 10,
        number_cycle_of_bias_update: int = 4,
        time_between_lambda_change: unit.Quantity = 10 * unit.picosecond,
    ) -> dict:
        """Run Gibbs Lambda Dynamics."""
        self.lambdas = lambdas
        self.lambdas_counts = np.zeros(len(self.lambdas))
        self.current_biases = np.zeros(len(self.lambdas))
        self.lambdas_sampled = []

        total_number_of_energy_evaluations = number_swap_before_bias_update * number_cycle_of_bias_update
        energies = np.zeros((total_number_of_energy_evaluations, len(lambdas)))

        # Start from a random lambda state
        self.current_lambda = np.random.choice(lambdas)
        self._update_lambda(self.current_lambda)

        number_md_steps_between_lambda_change = int(time_between_lambda_change / self.simulation.integrator.getStepSize())

        for bias_cycle in range(number_cycle_of_bias_update):
            for swap_cycle in range(number_swap_before_bias_update):
                self.simulation.step(number_md_steps_between_lambda_change)

                energy_this_step = self._compute_energies()
                energy_index = swap_cycle + bias_cycle * number_swap_before_bias_update
                energies[energy_index, :] = energy_this_step
                self.lambdas_sampled.append(self.current_lambda)
                self.lambdas_counts[np.where([self.current_lambda == lbd for lbd in self.lambdas])[0][0]] += 1
                print(energy_index)

                new_lambda = self._select_new_lambda(energy_this_step)
                self._update_lambda(new_lambda)
            self._update_biases(energies[: (energy_index + 1), :])

        mbar_object = self._run_mbar(energies)
        final_free_energies = mbar_object.compute_free_energy_differences()["Delta_f"][0] * self.kbt
        return {
            "free_energy_differences": final_free_energies,
            "lambda_counts": self.lambdas_counts,
            "free_energy_convergence": [
                ((i + 1) * time_between_lambda_change, energy) for i, energy in enumerate(np.array(self.free_energy_convergence))
            ],
            "mbar_object": mbar_object,
        }
