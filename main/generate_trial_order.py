import argparse
import json
import random

import numpy as np

from trial_order import (
    build_generation_config_for_selection,
    generate_trials_from_config,
    resolve_trial_order_selection,
    save_trial_order,
    trial_order_paths_from_experiment_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one-time randomized trial order for all subjects."
    )
    parser.add_argument(
        "--config",
        default="main/experiment_config.json",
        help="Path to experiment configuration JSON.",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "assist_first", "resist_first"],
        default="both",
        help="Generate both sequences, or only one selected sequence.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path when using --mode assist_first or --mode resist_first.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible regeneration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        experiment_config = json.load(f)

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    configured_paths = trial_order_paths_from_experiment_config(experiment_config)
    if args.mode == "both":
        selections = ["assist_first", "resist_first"]
    else:
        selections = [args.mode]

    for selection in selections:
        _, trial_randomization = resolve_trial_order_selection(selection)
        generation_config = build_generation_config_for_selection(experiment_config, selection)
        trials = generate_trials_from_config(generation_config)
        output_path = args.output if (args.output and len(selections) == 1) else configured_paths[selection]
        save_trial_order(output_path, generation_config, trials)
        print(f"Saved {selection} ({trial_randomization}) -> {output_path}")
        print(f"Total trials: {trials.shape[0]}")

    print("Use experiment.selected_trial_order to choose which sequence is loaded.")


if __name__ == "__main__":
    main()
