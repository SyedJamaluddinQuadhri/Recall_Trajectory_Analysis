from .activation_patching import (
    # Mode 1 — single-layer MLP (existing baseline)
    ActivationPatcher,
    run_patching_experiment,
    run_specificity_check,
    run_noise_control_check,
    # Mode 2 — window MLP (new)
    run_window_patching_experiment,
    run_window_specificity_check,
    run_window_noise_control,
    # Mode 3 — residual stream (new)
    run_residual_stream_patching_experiment,
    run_residual_specificity_check,
    run_residual_noise_control,
)
