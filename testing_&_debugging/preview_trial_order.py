import json
import sys

import numpy as np

sys.path.append("main")
from trial_order import load_trial_order

np.set_printoptions(threshold=sys.maxsize, linewidth=240)


def _build_block_info(generation_config: dict) -> tuple[bool, int]:
    trial_randomization = generation_config["trial_randomization"]
    trial_conditions = generation_config["trial_conditions"]

    resist_count = sum(c["condition_trial_no"] for c in trial_conditions if c["assistance"] == "resist")
    assist_count = sum(c["condition_trial_no"] for c in trial_conditions if c["assistance"] == "assist")
    transparent_count = sum(c["condition_trial_no"] for c in trial_conditions if c["assistance"] == "transparent")

    if trial_randomization == "resist_then_assist_with_transparent":
        return True, resist_count + transparent_count
    if trial_randomization == "assist_then_resist_with_transparent":
        return True, assist_count + transparent_count
    return False, 0


def main():
    with open("main/experiment_config.json", "r") as file:
        config = json.load(file)

    experiment = config["experiment"]
    selected = experiment.get("selected_trial_order", "assist_first")
    precomputed_paths = experiment.get("precomputed_trial_order_paths", {})
    trial_order_path = precomputed_paths.get(selected)
    if not trial_order_path:
        raise ValueError(
            f"No trial-order path for selected_trial_order='{selected}'. "
            "Check experiment.precomputed_trial_order_paths in config."
        )
    metadata, trials = load_trial_order(trial_order_path)
    generation_config = metadata["generation_config"]

    condition_names = {
        0: "resist",
        1: "assist",
        2: "transparent",
        99: "control",
    }
    event_names = {
        0: "DOWN",
        1: "UP",
    }

    familiarization_n = generation_config["familiarization_trials_no"]
    end_control_n = generation_config["end_control_trials_no"]
    main_start = familiarization_n
    main_end = len(trials) - end_control_n
    main_trials = trials[main_start:main_end]
    has_two_blocks, block_one_size = _build_block_info(generation_config)

    print("Columns: [global_index, phase, block, trial, condition, event, correctness, profile, torque]")
    print(f"Selected order: {selected}")
    print(f"Loaded from: {trial_order_path}")
    print(f"Randomization mode: {generation_config['trial_randomization']}")
    print(f"Total trials: {len(trials)}")
    print(f"Main trials: {len(main_trials)}")
    print()

    rows = []
    for index, trial in enumerate(trials, start=1):
        event, correctness, profile, torque, condition = trial
        condition = int(condition)

        if index <= familiarization_n:
            phase = "familiarization"
            block = 0
            trial_number = index
        elif index > main_end:
            phase = "end_control"
            block = 0
            trial_number = index - main_end
        else:
            phase = "main"
            main_index = index - main_start
            if has_two_blocks:
                block = 1 if main_index <= block_one_size else 2
            else:
                block = 1
            trial_number = main_index

        rows.append([
            index,
            phase,
            block,
            trial_number,
            condition_names.get(condition, "unknown"),
            event_names.get(int(event), "NA"),
            int(correctness),
            int(profile),
            float(torque),
        ])

    print(np.array(rows, dtype=object))


if __name__ == "__main__":
    main()
