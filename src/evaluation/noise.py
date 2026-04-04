"""Fixed noise model for all experiments."""

from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

# Fixed noise parameters — do not change between experiments
SINGLE_GATE_ERROR = 0.01
TWO_GATE_ERROR = 0.05
READOUT_ERROR_RATE = 0.03


def make_noise_model(single_gate_error: float = SINGLE_GATE_ERROR,
                     two_gate_error: float = TWO_GATE_ERROR,
                     readout_error_rate: float = READOUT_ERROR_RATE) -> NoiseModel:
    """
    Depolarizing noise model with readout error.

    Default rates (elevated but not absurd):
      1q gate depolarizing: 1%
      2q gate depolarizing: 5%
      Readout error: 3%
    """
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(single_gate_error, 1),
                                   ["rx", "rz", "h"])
    nm.add_all_qubit_quantum_error(depolarizing_error(two_gate_error, 2), ["cx"])
    if readout_error_rate > 0:
        re = ReadoutError([[1 - readout_error_rate, readout_error_rate],
                           [readout_error_rate, 1 - readout_error_rate]])
        nm.add_all_qubit_readout_error(re)
    return nm
