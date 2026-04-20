import json
import pygame
import threading
import logging
import argparse

from experiment_interface import Interface
from experiment_state_machine import StateMachine
from experiment_logging import Logger
from experiment_LSL import LSLHandler
from trial_order import (
    DEFAULT_TRIAL_ORDER_PATH,
    resolve_trial_order_selection,
)

def initialize_state_dict(state_dict, experiment_config):
    """
    Read JSON configuration file and create state_dict for saving current program parameters
    """
    if state_dict is None:
        state_dict = {}
        state_dict["state_wait_time"] = -1

    # Initialize state_dict with values from experiment_config
    state_dict["real_time_prediction"] = True if experiment_config["experiment"]["real_time_classifier_prediction"] == 1 else False
    state_dict["trial_states"] = experiment_config["experiment"]["define_trial_states"]
    state_dict["start_time_range"] = experiment_config["experiment"]["start_time_range"]
    state_dict["state_wait_time_range"] = experiment_config["experiment"]["state_wait_time_range"]
    state_dict["imagination_time_range"] = experiment_config["experiment"]["imagination_time_range"]
    state_dict["intention_time_range"] = experiment_config["experiment"]["intention_time_range"]
    state_dict["timeout"] = state_dict["TO"] = experiment_config["experiment"]["trial_timeout"]
    state_dict["familiarization_trials_No"] = experiment_config["experiment"]["number_of_familiarization_trials"]
    state_dict["end_control_trials"] = experiment_config["experiment"]["number_of_end_control_trials"]
    state_dict["trial_conditions"] = experiment_config["experiment"]["trial_conditions"]
    state_dict["randomize_trials"] = experiment_config["experiment"]["randomize_trials"]
    state_dict["selected_trial_order"] = experiment_config["experiment"].get("selected_trial_order", "assist_first")
    _, selected_trial_randomization = resolve_trial_order_selection(state_dict["selected_trial_order"])
    state_dict["trial_randomization"] = selected_trial_randomization

    precomputed_paths = experiment_config["experiment"].get("precomputed_trial_order_paths")
    if precomputed_paths:
        selected_path = precomputed_paths.get(state_dict["selected_trial_order"])
        if selected_path is None:
            raise ValueError(
                f"Missing precomputed_trial_order_paths['{state_dict['selected_trial_order']}'] in experiment_config."
            )
        state_dict["precomputed_trial_order_path"] = selected_path
    else:
        state_dict["precomputed_trial_order_path"] = experiment_config["experiment"].get(
            "precomputed_trial_order_path",
            DEFAULT_TRIAL_ORDER_PATH,
        )

    state_dict["fullscreen"] = experiment_config["interface_data"]["full_screen_mode"]
    state_dict["data_stream_interval"] = experiment_config["interface_data"]["data_stream_interval"]

    state_dict["exo_parameters"] = experiment_config["exo_parameters"]

    state_dict["enter_pressed"] = False
    state_dict["escape_pressed"] = False
    state_dict["space_pressed"] = False
    state_dict["stream_online"] = True
    state_dict["current_state"] = None
    state_dict["torque_profile"] = "None"
    state_dict["torque_magnitude"] = "None"
    state_dict["exo_condition"] = "None"
    
    prediction_stream = False
    if state_dict["real_time_prediction"]:
        state_dict["prediction"] = None
        prediction_stream = True

    # Initialize state machine parameters
    state_dict["trial"] = ""
    state_dict["event_id"] = 99
    state_dict["event_type"] = ""
    state_dict["current_trial_No"] = 0
    state_dict["remaining_time"] = ""
    state_dict["avg_time"] = None
    state_dict["familiarization_trial_No"] = 0
    state_dict["trials_No"] = 0

    state_dict["current_position"] = 0
    state_dict["current_velocity"] = 0
    state_dict["current_torque"] = 0
    state_dict["demanded_torque"] = 0
    state_dict["desired_torque"] = 0
    state_dict["measured_torque"] = 0

    state_dict["activate_EXO"] = False

    state_dict["background_color"] = "black"
    state_dict["color"] = "white"
    
    return state_dict, prediction_stream

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_log", action="store_true", help="Disable logging")
    args = parser.parse_args()

    # Load experiment configuration from JSON file
    experiment_config = json.load(open(r"main\experiment_config.json", "r"))
    
    # Setup logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Main")

    # Setup results logging
    save_data = True if experiment_config["interface_data"]["save_data"] == 1 else False
    data_log = Logger(
        experiment_config["interface_data"]["results_path"],    # results path
        experiment_config["participant"]["name"],               # participant name
        experiment_config["participant"]["id"],                 # participant ID
        args.no_log,                                            # disable logging
        save_data                                               # save data
    )
    data_log.save_experiment_config(experiment_config)

    # Initialize experiment state
    state_dict = None
    state_dict, predict = initialize_state_dict(state_dict, experiment_config)

    # Create an Inlet for incoming LSL Stream
    LSL = LSLHandler(state_dict, predict=predict)
    interface = Interface(
        state_dict  =   state_dict,
        maxP        =   state_dict["exo_parameters"]["maximum_arm_position_deg"],
        minP        =   state_dict["exo_parameters"]["minimum_arm_position_deg"]
    )
    state_machine = StateMachine(LSL)

    # Create a background thread for sending Data through LSL Stream
    stop_event = threading.Event()
    the_lock = threading.Lock()
    streamer_thread = threading.Thread(
        target=LSL.stream_events_data,
        args=(stop_event, state_dict, the_lock),
        daemon=True
    )
    streamer_thread.start()
    if state_dict["real_time_prediction"]:
        prediction_thread = threading.Thread(
            target=LSL.get_predictions,
            args=(stop_event, state_dict, True),
            daemon=True
        )
        prediction_thread.start()
    continue_experiment = True
    experiment_over = False

    try:
        while continue_experiment and experiment_over is False:
            pygame.event.clear()
            with the_lock:
                state_dict["timestamp"] = LSL.timestamp_g

            # Stream data and update state
            LSL.EXO_stream_in(state_dict)
            experiment_over, state_dict = state_machine.maybe_update_state(state_dict)
            continue_experiment = Interface.run(interface, state_dict)
            
            if "previous_state" not in state_dict:
                state_dict["previous_state"] = None
            if "event_type" not in state_dict:
                state_dict["event_type"] = None                     

            # Save data
            data_log.save_data_dict(state_dict)

    except Exception as e:
        logger.error(f"An error occurred during the experiment loop: {e}", exc_info=True)
    
    except KeyboardInterrupt:
        stop_event.set()
        continue_experiment = False

    finally:
        data_log.close()
