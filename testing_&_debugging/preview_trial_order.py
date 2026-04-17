import json
import sys

import numpy as np

sys.path.append("main")
from experiment_state_machine import StateMachine

np.set_printoptions(threshold=sys.maxsize, linewidth=240)


def main():
    with open("main/experiment_config.json", "r") as file:
        config = json.load(file)

    state = {
        "trial_conditions": config["experiment"]["trial_conditions"],
        "familiarization_trials_No": config["experiment"]["number_of_familiarization_trials"],
        "end_control_trials": config["experiment"]["number_of_end_control_trials"],
        "randomize_trials": config["experiment"]["randomize_trials"],
        "trial_randomization": config["experiment"].get("trial_randomization", "full_random"),
        "exo_parameters": config["exo_parameters"],
    }

    state_machine = StateMachine(None)
    trials = state_machine.generate_trials(state)

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

    familiarization_n = state["familiarization_trial_No"]
    main_start = familiarization_n
    main_end = len(trials) - state["end_control_trials"]
    main_trials = trials[main_start:main_end]

    print("Columns: [global_index, phase, block, trial, condition, event, correctness, profile, torque]")
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
            block = 1 if main_index <= 150 else 2
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
