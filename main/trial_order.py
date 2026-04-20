import hashlib
import json
import random
from pathlib import Path

import numpy as np

TRIAL_ORDER_FORMAT_VERSION = 1
TRIAL_ORDER_MAGIC = "EDUEXO_TRIAL_ORDER_V1"
DEFAULT_TRIAL_ORDER_PATH = "main/trial_order_sequence.txt"
DEFAULT_ASSIST_FIRST_TRIAL_ORDER_PATH = "main/trial_order_assist_first.txt"
DEFAULT_RESIST_FIRST_TRIAL_ORDER_PATH = "main/trial_order_resist_first.txt"

TRIAL_ORDER_SELECTION_TO_RANDOMIZATION = {
    "assist_first": "assist_then_resist_with_transparent",
    "resist_first": "resist_then_assist_with_transparent",
}

PROFILES_DICT = {
    "trapezoid": 0,
    "triangular": 1,
    "sinusoidal": 2,
    "rectangular": 3,
    "smooth_trapezoid": 4,
}


def normalize_trial_conditions(trial_conditions: dict) -> list:
    normalized = []
    for condition_id, values in sorted(trial_conditions.items(), key=lambda kv: int(kv[0])):
        assistance, condition_trial_no, torque_profile, torque = values
        normalized.append(
            {
                "condition_id": str(condition_id),
                "assistance": str(assistance),
                "condition_trial_no": int(condition_trial_no),
                "torque_profile": str(torque_profile),
                "torque": float(torque),
            }
        )
    return normalized


def build_generation_config_from_experiment_config(experiment_config: dict) -> dict:
    experiment = experiment_config["experiment"]
    exo_parameters = experiment_config["exo_parameters"]
    return {
        "familiarization_trials_no": int(experiment["number_of_familiarization_trials"]),
        "end_control_trials_no": int(experiment["number_of_end_control_trials"]),
        "trial_conditions": normalize_trial_conditions(experiment["trial_conditions"]),
        "randomize_trials": int(experiment["randomize_trials"]),
        "trial_randomization": str(experiment.get("trial_randomization", "full_random")),
        "torque_limit": float(exo_parameters["torque_limit"]),
    }


def build_generation_config_for_selection(experiment_config: dict, selection: str) -> dict:
    base_config = build_generation_config_from_experiment_config(experiment_config)
    _, trial_randomization = resolve_trial_order_selection(selection)
    base_config["trial_randomization"] = trial_randomization
    return base_config


def build_generation_config_from_state_dict(state_dict: dict) -> dict:
    return {
        "familiarization_trials_no": int(state_dict["familiarization_trials_No"]),
        "end_control_trials_no": int(state_dict["end_control_trials"]),
        "trial_conditions": normalize_trial_conditions(state_dict["trial_conditions"]),
        "randomize_trials": int(state_dict["randomize_trials"]),
        "trial_randomization": str(state_dict.get("trial_randomization", "full_random")),
        "torque_limit": float(state_dict["exo_parameters"]["torque_limit"]),
    }


def trial_order_path_from_experiment_config(experiment_config: dict) -> str:
    return experiment_config["experiment"].get("precomputed_trial_order_path", DEFAULT_TRIAL_ORDER_PATH)


def trial_order_paths_from_experiment_config(experiment_config: dict) -> dict:
    experiment = experiment_config["experiment"]
    configured = experiment.get("precomputed_trial_order_paths", {})
    return {
        "assist_first": configured.get("assist_first", DEFAULT_ASSIST_FIRST_TRIAL_ORDER_PATH),
        "resist_first": configured.get("resist_first", DEFAULT_RESIST_FIRST_TRIAL_ORDER_PATH),
    }


def resolve_trial_order_selection(selection: str) -> tuple[str, str]:
    selection_key = str(selection)
    if selection_key not in TRIAL_ORDER_SELECTION_TO_RANDOMIZATION:
        valid = ", ".join(sorted(TRIAL_ORDER_SELECTION_TO_RANDOMIZATION.keys()))
        raise ValueError(f"Invalid selected_trial_order '{selection_key}'. Expected one of: {valid}")
    return selection_key, TRIAL_ORDER_SELECTION_TO_RANDOMIZATION[selection_key]


def _generate_familiarization_trials(n: int) -> np.ndarray:
    ones = np.ones(n // 2, dtype=int)
    zeros = np.zeros(n // 2, dtype=int)
    if n % 2 != 0:
        ones = np.append(ones, 1)
    events = np.append(ones, zeros)
    executions = torque_profiles = torques = exo_conditions = 99 * np.ones(n, dtype=int)
    trials = np.column_stack((events, executions, torque_profiles, torques, exo_conditions))
    np.random.shuffle(trials)
    return trials


def _generate_trials_for_condition(condition: dict, torque_limit: float) -> np.ndarray:
    assistance = condition["assistance"]
    condition_trial_no = int(condition["condition_trial_no"])
    torque_profile = condition["torque_profile"]
    torque_magnitude = float(condition["torque"])

    ones = np.ones(condition_trial_no // 2, dtype=int)
    zeros = np.zeros(condition_trial_no // 2, dtype=int)
    events = np.append(ones, zeros)
    if condition_trial_no % 2 != 0:
        events = np.append(events, int(1))

    if assistance == "assist":
        executions = np.ones(condition_trial_no, dtype=int)
        exo_conditions = np.ones(condition_trial_no, dtype=int)
    elif assistance == "resist":
        executions = np.zeros(condition_trial_no, dtype=int)
        exo_conditions = np.zeros(condition_trial_no, dtype=int)
    elif assistance == "transparent":
        executions = np.ones(condition_trial_no, dtype=int)
        exo_conditions = 2 * np.ones(condition_trial_no, dtype=int)
        torque_magnitude = 0
    else:
        raise ValueError(f"Invalid assistance type in condition {condition['condition_id']}: {assistance}")

    if torque_profile == "random":
        torque_profile_labels = [random.choice(list(PROFILES_DICT.values())) for _ in range(condition_trial_no)]
    elif torque_profile in PROFILES_DICT:
        torque_profile_labels = [PROFILES_DICT[torque_profile]] * condition_trial_no
    else:
        raise ValueError(f"Invalid torque profile in condition {condition['condition_id']}: {torque_profile}")

    if torque_magnitude > torque_limit:
        torque_magnitude = torque_limit

    torque_magnitude_labels = [torque_magnitude] * condition_trial_no

    condition_trials = np.column_stack(
        (
            events.astype(int),
            executions.astype(int),
            np.asarray(torque_profile_labels, dtype=int),
            np.asarray(torque_magnitude_labels, dtype=float),
            exo_conditions.astype(int),
        )
    )
    np.random.shuffle(condition_trials)
    return condition_trials


def _stack_condition_trials(condition_trials: list) -> np.ndarray:
    if condition_trials:
        return np.vstack(condition_trials)
    return np.empty((0, 5))


def _shuffle_if_needed(trials: np.ndarray, randomize_trials: int) -> np.ndarray:
    if int(randomize_trials):
        np.random.shuffle(trials)
    return trials


def generate_trials_from_config(generation_config: dict) -> np.ndarray:
    familiarization_trials = _generate_familiarization_trials(generation_config["familiarization_trials_no"])
    end_control_no = generation_config["end_control_trials_no"]
    end_control_trials = _generate_familiarization_trials(end_control_no) if end_control_no else None

    condition_trials_by_exo_condition = {0: [], 1: [], 2: []}
    for condition in generation_config["trial_conditions"]:
        condition_trials = _generate_trials_for_condition(condition, generation_config["torque_limit"])
        exo_condition = int(condition_trials[0, 4])
        condition_trials_by_exo_condition[exo_condition].append(condition_trials)

    randomize_trials = generation_config["randomize_trials"]
    trial_randomization = generation_config["trial_randomization"]
    resist_trials = _stack_condition_trials(condition_trials_by_exo_condition[0])
    assist_trials = _stack_condition_trials(condition_trials_by_exo_condition[1])
    transparent_trials = _stack_condition_trials(condition_trials_by_exo_condition[2])

    if trial_randomization == "resist_then_assist_with_transparent":
        resist_block = _shuffle_if_needed(np.vstack((resist_trials, transparent_trials)), randomize_trials)
        assist_block = _shuffle_if_needed(np.vstack((assist_trials, transparent_trials)), randomize_trials)
        main_trials = np.vstack((resist_block, assist_block))
    elif trial_randomization == "assist_then_resist_with_transparent":
        assist_block = _shuffle_if_needed(np.vstack((assist_trials, transparent_trials)), randomize_trials)
        resist_block = _shuffle_if_needed(np.vstack((resist_trials, transparent_trials)), randomize_trials)
        main_trials = np.vstack((assist_block, resist_block))
    else:
        all_condition_trials = []
        for condition_trials in condition_trials_by_exo_condition.values():
            all_condition_trials.extend(condition_trials)
        main_trials = _shuffle_if_needed(np.vstack(all_condition_trials), randomize_trials)

    if end_control_trials is not None:
        final_trials = np.vstack((familiarization_trials, main_trials, end_control_trials))
    else:
        final_trials = np.vstack((familiarization_trials, main_trials))

    return final_trials


def _trial_matrix_checksum(trials: np.ndarray) -> str:
    trials_for_hash = np.asarray(trials, dtype=np.float64)
    return hashlib.sha256(trials_for_hash.tobytes()).hexdigest()


def expected_total_trial_count(generation_config: dict) -> int:
    fam = int(generation_config["familiarization_trials_no"])
    end_ctrl = int(generation_config["end_control_trials_no"])
    randomization = generation_config["trial_randomization"]

    resist_count = 0
    assist_count = 0
    transparent_count = 0
    other_count = 0
    for condition in generation_config["trial_conditions"]:
        count = int(condition["condition_trial_no"])
        assistance = condition["assistance"]
        if assistance == "resist":
            resist_count += count
        elif assistance == "assist":
            assist_count += count
        elif assistance == "transparent":
            transparent_count += count
        else:
            other_count += count

    if randomization in {"resist_then_assist_with_transparent", "assist_then_resist_with_transparent"}:
        main_trials = resist_count + assist_count + 2 * transparent_count
    else:
        main_trials = resist_count + assist_count + transparent_count + other_count

    return fam + main_trials + end_ctrl


def save_trial_order(path: str, generation_config: dict, trials: np.ndarray) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format_version": TRIAL_ORDER_FORMAT_VERSION,
        "magic": TRIAL_ORDER_MAGIC,
        "generation_config": generation_config,
        "trial_count": int(trials.shape[0]),
        "trial_matrix_sha256": _trial_matrix_checksum(trials),
    }

    with path_obj.open("w", encoding="utf-8") as f:
        f.write(f"# {TRIAL_ORDER_MAGIC}\n")
        f.write(f"# metadata_json={json.dumps(metadata, sort_keys=True)}\n")
        f.write("event,execution,torque_profile,torque_magnitude,exo_condition\n")
        np.savetxt(f, trials, delimiter=",", fmt="%.10g")


def load_trial_order(path: str) -> tuple[dict, np.ndarray]:
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError(f"Trial order file is too short: {path}")
    if not lines[0].strip() == f"# {TRIAL_ORDER_MAGIC}":
        raise ValueError(f"Invalid trial order magic header in {path}")
    if not lines[1].startswith("# metadata_json="):
        raise ValueError(f"Missing trial order metadata line in {path}")

    metadata = json.loads(lines[1][len("# metadata_json=") :].strip())

    trials = np.loadtxt(path, delimiter=",", skiprows=3)
    if trials.ndim == 1:
        trials = np.expand_dims(trials, axis=0)
    return metadata, trials


def validate_trial_order_for_state_dict(metadata: dict, trials: np.ndarray, state_dict: dict) -> None:
    if metadata.get("magic") != TRIAL_ORDER_MAGIC:
        raise ValueError("Trial order file has invalid magic identifier.")
    if int(metadata.get("format_version", -1)) != TRIAL_ORDER_FORMAT_VERSION:
        raise ValueError("Trial order file format version is not supported.")

    expected_generation_config = build_generation_config_from_state_dict(state_dict)
    file_generation_config = metadata.get("generation_config")
    if file_generation_config != expected_generation_config:
        raise ValueError(
            "Trial order file is incompatible with current experiment config. "
            "Regenerate trial order for this configuration."
        )

    declared_trial_count = int(metadata.get("trial_count", -1))
    actual_trial_count = int(trials.shape[0])
    if declared_trial_count != actual_trial_count:
        raise ValueError(
            f"Trial order file corrupted: metadata trial_count={declared_trial_count}, "
            f"actual rows={actual_trial_count}."
        )

    expected_trial_count = expected_total_trial_count(expected_generation_config)
    if expected_trial_count != actual_trial_count:
        raise ValueError(
            f"Trial order row count mismatch: expected {expected_trial_count}, got {actual_trial_count}."
        )

    file_checksum = metadata.get("trial_matrix_sha256")
    computed_checksum = _trial_matrix_checksum(trials)
    if file_checksum != computed_checksum:
        raise ValueError("Trial order checksum mismatch. File may be modified or corrupted.")


def apply_trials_to_state_machine(state_machine, state_dict: dict, trials: np.ndarray) -> None:
    state_dict["familiarization_trial_No"] = int(state_dict["familiarization_trials_No"])
    state_dict["trials_No"] = int(trials.shape[0])
    state_machine.events = trials[:, 0]
    state_machine.correctness_list = trials[:, 1]
    state_machine.torque_profile_list = trials[:, 2]
    state_machine.torque_magnitude_list = trials[:, 3]
    state_machine.exo_condition_list = trials[:, 4]
