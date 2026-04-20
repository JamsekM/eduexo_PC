from time import time
import numpy as np
import random
import pygame
import logging
from trial_order import (
    apply_trials_to_state_machine,
    load_trial_order,
    validate_trial_order_for_state_dict,
)

class StateMachine:
    """ 
    EVENT IDS vs EVENT TYPES:
        99 - no_event
        10 - imagine_UP (prompt "UP" shown, RED)
        11 - intend_UP (prompt "UP" changed, Yellow)
        12 - execute_UP (prompt "UP" changed, Green)
        13 - moving_UP
        20 - imagine_DOWN (prompt "DOWN" shown, RED)
        21 - intend_DOWN (prompt "DOWN" changed, Yellow)
        22 - execute_DOWN (prompt "DOWN" changed, Green)
        23 - moving_DOWN
        30 - exo_execution_correct
        40 - exo_execution_incorrect
        50 - success
        60 - failure
        70 - timeout
        0 - INITIAL_SCREEN
    """
    
    no_event                    = 99
    imagine_UP                  = 10
    intend_UP                   = 11
    execute_UP                  = 12
    moving_UP                  = 13
    imagine_DOWN                = 20
    intend_DOWN                 = 21
    execute_DOWN                = 22
    moving_DOWN                = 23
    exo_execution_correct       = 30
    exo_execution_incorrect     = 40
    success                     = 50
    failure                     = 60
    timeout                     = 70
    
    # State IDs
    INITIAL_SCREEN = 0


    RETURN_TO_CENTER = 1
    IN_MIDDLE_CIRCLE = 2
    WAITING = 3
    IMAGINATION = 4
    INTENTION = 5

    TRIAL_UP = 6
    MOVING_UP = 7
    IN_UPPER_BAND = 8
    GO_OUT_OF_UPPER_BAND = 9
    TRIAL_DOWN = 10
    MOVING_DOWN = 11
    IN_LOWER_BAND = 12
    GO_OUT_OF_LOWER_BAND = 13

    FAILURE = 14
    TIMEOUT = 15
    PAUSE = 16
    EXIT = 17

    def __init__(self, LSL):
        self.current_state = None
        self.torque = None
        self.torque_profile = None
        self.correctness = None
        self.exo_condition = None
        self.LSL = LSL
        self.i = 0
        self.times = []
        self.send_once = True      
        self.stream_break = False  
        # Construct reverse state lookup
        all_variables = vars(StateMachine)
        self.reverse_state_lookup = {all_variables[name]: name for name in all_variables if isinstance(all_variables[name], int) and name.isupper()}
        self.profiles_dict = {"trapezoid" : 0, "triangular" : 1, "sinusoidal" : 2, "rectangular" : 3, "smooth_trapezoid" : 4}

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("state_machine")
        self.logger = logger

    
    def maybe_update_state(self, state_dict: dict):
        """
        Updates the state of the experiment state machine based on the current state and state dictionary.
        Args:
            state_dict (dict): A dictionary containing the current state information and various flags.
        Returns:
            tuple: A tuple containing a boolean indicating if the experiment is over and the updated state dictionary.
        The state machine transitions through the following states:
            - INITIAL_SCREEN: Initial screen state.
            - RETURN_TO_CENTER: Waiting for the initial position.
            - IN_MIDDLE_CIRCLE: Waiting in the middle for the trial to start.
            - WAITING: Waiting for the start of the trial.
            - IMAGINATION: Imagining movement.
            - INTENTION: Intending movement.
            - TRIAL_UP: Executing "UP" trial.
            - MOVING_UP: Moving up.
            - IN_UPPER_BAND: "UP" trial successful.
            - TRIAL_DOWN: Executing "DOWN" trial.
            - MOVING_DOWN: Moving down.
            - IN_LOWER_BAND: "DOWN" trial successful.
            - TIMEOUT: Timeout occurred.
            - FAILURE: Failure occurred.
            - PAUSE: Experiment paused.
            - EXIT: Exiting the experiment.
        The function handles various events and transitions, including:
            - Enter key press to start the experiment.
            - Space key press to pause/unpause the experiment.
            - Escape key press to terminate the experiment.
            - Stream online/offline status.
            - Trial execution and success/failure handling.
            - Timeout handling.
        """

        experiment_over = False  

        # --- Handle keyboard input for state transitions ---
        self.one_time_ENTER(state_dict)  # One time ENTER trigger
        self.one_time_ESCAPE(state_dict)  # One time ESC trigger
        self.latch_SPACE(state_dict)  # SPACE latch

        #### WHEN NONE
        # If no state is set, initialize to the initial screen
        if self.current_state is None:  
            self.current_state = StateMachine.INITIAL_SCREEN
            self.set_initial_screen(state_dict)

        #### AT THE BEGINNING
        # Transition from initial screen to experiment start when ENTER is pressed
        elif self.current_state == StateMachine.INITIAL_SCREEN:
            if state_dict["enter_pressed"]:
                self.current_state = StateMachine.RETURN_TO_CENTER
                self.set_start_experiment(state_dict)
                self.set_return_to_center(state_dict)

        #### WHEN WAITING FOR INITIAL POSITION
        # Wait for subject to move to the middle position before starting trials
        elif self.current_state == StateMachine.RETURN_TO_CENTER:
            if state_dict["in_the_middle"]:
                self.current_state = StateMachine.IN_MIDDLE_CIRCLE
                self.set_in_middle_circle(state_dict)

        #### WHEN WAITING IN THE MIDDLE FOR TRIAL TO START
        # Wait for random time, then start a new trial if ready
        elif self.current_state == StateMachine.IN_MIDDLE_CIRCLE:
            if state_dict["in_the_middle"] == False:
                self.current_state = StateMachine.RETURN_TO_CENTER
                self.set_return_to_center(state_dict)
            elif time() - state_dict["state_start_time"] >= state_dict["state_wait_time"]:
                self.send_once = True
                if self.i < len(self.events):
                    # Set up next trial parameters
                    self.correctness = self.correctness_list[self.i]
                    self.torque_profile = self.torque_profile_list[self.i]
                    self.torque = self.torque_magnitude_list[self.i]
                    self.exo_condition = self.exo_condition_list[self.i]
                    state_dict["exo_condition"] = self.exo_condition_name(self.exo_condition)
                    if self.events[self.i] == 1:
                        self.i += 1
                        state_dict["trial"] = "UP"
                        state_dict["current_trial_No"] = self.i
                    else:
                        self.i += 1
                        state_dict["trial"] = "DOWN"
                        state_dict["current_trial_No"] = self.i
                    # Determine which state to enter next based on trial_states
                    if "wait" in state_dict["trial_states"]:
                        self.current_state = StateMachine.WAITING
                        self.set_waiting(state_dict)
                    else:
                        if "imagine" in state_dict["trial_states"]:
                            self.set_imagine(state_dict)
                        else:
                            if "intend" in state_dict["trial_states"]:
                                self.set_intend(state_dict)
                            else:
                                self.set_go_to_band(state_dict)

        #### WHEN WAITING FOR START
        # Wait for a random time, then move to imagination/intention/execution
        elif self.current_state == StateMachine.WAITING:
            if time() - state_dict["state_start_time"] >= state_dict["state_wait_time"]:
                if "imagine" in state_dict["trial_states"]:
                    self.set_imagine(state_dict)
                else:
                    if "intend" in state_dict["trial_states"]:
                        self.set_intend(state_dict)
                    else:
                        self.set_go_to_band(state_dict)

        #### WHEN IMAGINING MOVEMENT
        # After imagination, move to intention or execution
        elif self.current_state == StateMachine.IMAGINATION:
            if time() - state_dict["state_start_time"] >= state_dict["state_wait_time"]:
                if "intend" in state_dict["trial_states"]:
                    self.set_intend(state_dict)
                else:
                    self.set_go_to_band(state_dict)

        #### WHEN INTENDING MOVEMENT
        # After intention, move to execution
        elif self.current_state == StateMachine.INTENTION:
            if time() - state_dict["state_start_time"] >= state_dict["state_wait_time"]:
                self.set_go_to_band(state_dict)

        #### WHEN "UP" TRIAL IS HAPPENING
        # Handle execution of "UP" trial, check for success/failure
        elif self.current_state == StateMachine.TRIAL_UP:
            if state_dict["in_the_middle"] == False:
                self.set_trial_start(state_dict)
                self.current_state = StateMachine.MOVING_UP
                if state_dict["activate_EXO"]:
                    if not state_dict["real_time_prediction"]:
                        if self.send_once:
                            self.LSL.EXO_stream_out(state_dict, self.torque_profile, self.torque, self.correctness)
                            self.send_once = False
                    else: 
                        if state_dict["prediction"] is not None:
                            self.LSL.EXO_stream_out(state_dict, self.torque_profile, self.torque)
                            state_dict["prediction"] = None

        elif self.current_state == StateMachine.MOVING_UP:
            if state_dict["is_UP"]:
                self.current_state = StateMachine.IN_UPPER_BAND
                self.set_in_upper_band(state_dict)
                self.set_success(state_dict)
                self.times.append(state_dict["TO"] - state_dict["remaining_time"])
            elif state_dict["is_DOWN"]:
                self.current_state = StateMachine.FAILURE
                self.set_failure(state_dict)  

        #### IF "UP" TRIAL IS SUCCESSFUL
        # After success, return to center or finish experiment
        elif self.current_state == StateMachine.IN_UPPER_BAND:
            if time() - state_dict["state_start_time"] >= state_dict["state_wait_time"]:
                self.current_state = StateMachine.RETURN_TO_CENTER
                self.set_return_to_center(state_dict)
                if self.i == len(self.events):
                    self.current_state = StateMachine.EXIT
                    self.set_exit_or_error(state_dict)
                    state_dict["avg_time"] = round(sum(self.times) / len(self.times), 2)
                    state_dict["succ_trials"] = len(self.times)

        #### WHEN "DOWN" TRIAL IS HAPPENING
        # Handle execution of "DOWN" trial, check for success/failure
        elif self.current_state == StateMachine.TRIAL_DOWN:
            if state_dict["in_the_middle"] == False:
                self.set_trial_start(state_dict)
                self.current_state = StateMachine.MOVING_DOWN
                if state_dict["activate_EXO"]:
                    if not state_dict["real_time_prediction"]:
                        if self.send_once:
                            self.LSL.EXO_stream_out(state_dict, self.torque_profile, self.torque, self.correctness)
                            self.send_once = False
                    else: 
                        if state_dict["prediction"] is not None:
                            self.LSL.EXO_stream_out(state_dict, self.torque_profile, self.torque)
                            state_dict["prediction"] = None
            
        elif self.current_state == StateMachine.MOVING_DOWN:    
            if state_dict["is_DOWN"]:
                self.current_state = StateMachine.IN_LOWER_BAND
                self.set_in_lower_band(state_dict)
                self.set_success(state_dict)
                self.times.append(state_dict["TO"] - state_dict["remaining_time"])
            elif state_dict["is_UP"]:
                self.current_state = StateMachine.FAILURE
                self.set_failure(state_dict)                              
        
        #### IF "DOWN" TRIAL IS SUCCESSFUL
        # After success, return to center or finish experiment
        elif self.current_state == StateMachine.IN_LOWER_BAND:
            if time() - state_dict["state_start_time"] >= state_dict["state_wait_time"]:
                self.current_state = StateMachine.RETURN_TO_CENTER
                self.set_return_to_center(state_dict)  
                if self.i == len(self.events):
                    self.current_state = StateMachine.EXIT
                    self.set_exit_or_error(state_dict)              
                    state_dict["avg_time"] = round(sum(self.times) / len(self.times), 2)
                    state_dict["succ_trials"] = len(self.times)

        #### WHEN TIMEOUT OR FAILURE OCCUR
        # Handle timeout or failure, return to center or finish experiment
        elif self.current_state in {StateMachine.TIMEOUT, StateMachine.FAILURE}:
            if time() - state_dict["state_start_time"] >= state_dict["state_wait_time"]:
                if self.i == len(self.events):
                    self.current_state = StateMachine.EXIT
                    self.set_exit_or_error(state_dict)
                    if self.times == []:
                        state_dict["avg_time"] = 0
                else:
                    self.current_state = StateMachine.RETURN_TO_CENTER
                    self.set_return_to_center(state_dict)

        #### PAUSE
        # Handle pause logic (SPACE key)
        if state_dict["space_pressed"] and not self.current_state == StateMachine.EXIT:
            if self.current_state != StateMachine.PAUSE:
                if state_dict["trial_in_progress"] and self.current_state in {StateMachine.MOVING_UP, StateMachine.MOVING_DOWN}:
                    state_dict["timeout"] = state_dict["remaining_time"]
                self.previous_trial = state_dict["trial"]
                self.previous_event_id = state_dict["event_id"]
                self.previous_event_type = state_dict["event_type"]
                state_dict["event_id"] = StateMachine.no_event
                state_dict["event_type"] = ""
                self.previous_state = self.current_state
                self.current_state = StateMachine.PAUSE
        
        #### UNPAUSE
        # Resume from pause, restore previous state
        elif self.current_state == StateMachine.PAUSE:
            if state_dict["space_pressed"] == False:
                if self.previous_state == "IN_MIDDLE_CIRCLE":
                    self.previous_state = StateMachine.RETURN_TO_CENTER
                    self.set_return_to_center(state_dict)
                else:    
                    self.current_state = self.previous_state
                state_dict["trial_time"] = time()
                state_dict["state_start_time"] = time()
                state_dict["trial"] = self.previous_trial
                state_dict["event_id"] = self.previous_event_id
                state_dict["event_type"] = self.previous_event_type
                state_dict["trial_in_progress"] = True

        #### WHEN EXITING
        # Handle experiment exit and restart logic
        if self.current_state == StateMachine.EXIT and state_dict["stream_online"] == True:
            if self.stream_break == True:
                self.logger.info("Reconnected!")
                self.stream_break = False
            if state_dict["escape_pressed"]:
                experiment_over = True
                self.LSL.EXO_stream_out(experiment_over=True)
            elif state_dict["enter_pressed"]:
                self.i = 0
                self.times = []
                self.current_state = None
                state_dict["space_pressed"] = False
                state_dict["needs_update"] = True
                state_dict["avg_time"] = None

        #### FORCE TERMINATION
        # Handle forced experiment termination (ESC key)
        elif state_dict["escape_pressed"] == True:
            self.current_state = StateMachine.EXIT
            self.set_exit_or_error(state_dict, "firebrick", "EXPERIMENT TERMINATED")

        #### STREAM BREAK 
        # Handle stream offline state
        if state_dict["stream_online"] == False:
            self.current_state = StateMachine.EXIT
            self.stream_break = True
            self.set_exit_or_error(state_dict, "firebrick", "STREAM OFFLINE", "Press ESC to exit or press ENTER when stream is online")

        #### WHILE TRIAL IS IN PROGRESS
        # Update trial timing and check for timeout
        if self.current_state in {StateMachine.WAITING, StateMachine.IMAGINATION, StateMachine.INTENTION, StateMachine.TRIAL_UP, StateMachine.TRIAL_DOWN, StateMachine.MOVING_UP, StateMachine.MOVING_DOWN}:
            state_dict["trial_in_progress"] = True
            if self.current_state in {StateMachine.MOVING_DOWN, StateMachine.MOVING_UP}:
                if state_dict["exo_execution"] == 1:
                    if not state_dict["real_time_prediction"]:
                        if self.correctness == 1:
                            state_dict["event_id"] = StateMachine.exo_execution_correct
                            state_dict["event_type"] = "exo_execution_correct"
                        else:
                            state_dict["event_id"] = StateMachine.exo_execution_incorrect
                            state_dict["event_type"] = "exo_execution_incorrect"

                state_dict["remaining_time"] = round(state_dict["timeout"] - (time() - state_dict["trial_time"]), 1)

                if state_dict["remaining_time"] <= 0:
                    self.current_state = StateMachine.TIMEOUT
                    self.set_trial_timeout(state_dict)
        else:
            # Reset trial-related fields when not in a trial
            if self.current_state not in {StateMachine.IN_UPPER_BAND, StateMachine.IN_LOWER_BAND}:
                state_dict["trial"] = ""
            state_dict["trial_in_progress"] = False
            state_dict["remaining_time"] = ""
            state_dict["torque_profile"] = "None"
            state_dict["torque_magnitude"] = "None"
            state_dict["exo_condition"] = "None"

        # Set current state name for display/logging
        if self.current_state is not None:
            state_dict["current_state"] = self.reverse_state_lookup[self.current_state]             
        else:
            state_dict["current_state"] = "None"

        # Enable exoskeleton only for main trials (not familiarization or end control)
        if state_dict["familiarization_trial_No"] < self.i <= state_dict["trials_No"] - state_dict["end_control_trials"]:
            state_dict["activate_EXO"] = True
        else:
            state_dict["activate_EXO"] = False

        # Reset event info when returning to center
        if self.current_state == StateMachine.RETURN_TO_CENTER:
            state_dict["event_id"] = StateMachine.no_event
            state_dict["event_type"] = ""

        return experiment_over, state_dict

    #### STATE SETTERS
    def set_initial_screen(self, state_dict): 
        state_dict["state_start_time"] = None
        state_dict["background_color"] = "green4"
        state_dict["current_trial_No"] = 0

    def set_start_experiment(self, state_dict):
        trial_order_path = state_dict["precomputed_trial_order_path"]
        try:
            metadata, trials = load_trial_order(trial_order_path)
            validate_trial_order_for_state_dict(metadata, trials, state_dict)
            apply_trials_to_state_machine(self, state_dict, trials)
        except Exception as e:
            self.logger.error(f"Failed to load compatible precomputed trial order from '{trial_order_path}': {e}")
            raise
        state_dict["experiment_start"] = time()
        state_dict["main_text"] = ""
        state_dict["background_color"] = "black"

    def set_return_to_center(self, state_dict):
        state_dict["state_start_time"] = None
        state_dict["main_text"] = ""
        state_dict["timeout"] = state_dict["TO"]

    def set_in_middle_circle(self, state_dict):
        state_dict["state_start_time"] = time()
        state_dict["state_wait_time"] = np.random.uniform(*state_dict["start_time_range"])  # s
        state_dict["main_text"] = ""
    
    def set_waiting(self, state_dict):
        state_dict["state_start_time"] = time()
        state_dict["state_wait_time"] = np.random.uniform(*state_dict["state_wait_time_range"])  # s
        state_dict["color"] = "white"

    def set_imagine(self, state_dict):
        self.current_state = StateMachine.IMAGINATION
        if state_dict["trial"] == "UP":
            state_dict["event_id"] = StateMachine.imagine_UP
            state_dict["event_type"] = "imagine_UP"
        else:
            state_dict["event_type"] = "imagine_DOWN"
            state_dict["event_id"] = StateMachine.imagine_DOWN

        state_dict["state_start_time"] = time()
        state_dict["state_wait_time"] = np.random.uniform(*state_dict["imagination_time_range"])  # s
        state_dict["color"] = "red"

    def set_intend(self, state_dict):
        self.current_state = StateMachine.INTENTION
        if state_dict["trial"] == "UP":
            state_dict["event_type"] = "intend_UP"
            state_dict["event_id"] = StateMachine.intend_UP
        else:
            state_dict["event_type"] = "intend_DOWN"
            state_dict["event_id"] = StateMachine.intend_DOWN

        state_dict["state_start_time"] = time()
        state_dict["state_wait_time"] = np.random.uniform(*state_dict["intention_time_range"])  # s
        state_dict["color"] = "yellow"

    def set_in_upper_band(self, state_dict):
        state_dict["state_start_time"] = time()
        state_dict["state_wait_time"] = 1
        state_dict["main_text"] = "SUCCESS"

    def set_in_lower_band(self, state_dict):
        state_dict["state_start_time"] = time()
        state_dict["state_wait_time"] = 1
        state_dict["main_text"] = "SUCCESS"

    def set_go_to_band(self, state_dict):
        if state_dict["trial"] == "UP":
            self.current_state = StateMachine.TRIAL_UP
            state_dict["event_type"] = "execute_UP"
            state_dict["event_id"] = StateMachine.execute_UP
        else:
            self.current_state = StateMachine.TRIAL_DOWN
            state_dict["event_type"] = "execute_DOWN"
            state_dict["event_id"] = StateMachine.execute_DOWN
        state_dict["color"] = "green3"

    def set_trial_start(self, state_dict):
        if state_dict["trial"] == "UP":
            self.current_state = StateMachine.TRIAL_UP
            state_dict["event_type"] = "moving_UP"
            state_dict["event_id"] = StateMachine.moving_UP
        else:
            self.current_state = StateMachine.TRIAL_DOWN
            state_dict["event_type"] = "moving_DOWN"
            state_dict["event_id"] = StateMachine.moving_DOWN
        state_dict["trial_time"] = time()
        
    def set_success(self, state_dict):
        state_dict["event_type"] = "SUCCESS"
        state_dict["event_id"] = StateMachine.success    
        self.LSL.EXO_stream_out(state_dict, trial_over = True)

    def set_failure(self, state_dict):
        state_dict["state_start_time"] = time()
        state_dict["state_wait_time"] = 1.5
        state_dict["main_text"] = state_dict["event_type"] = "FAIL"
        state_dict["event_id"] = StateMachine.failure  
        self.LSL.EXO_stream_out(state_dict, trial_over = True)

    def set_trial_timeout(self, state_dict):
        state_dict["state_start_time"] = time()
        state_dict["state_wait_time"] = 1.5
        state_dict["main_text"] = state_dict["event_type"] = "TIMEOUT"
        state_dict["event_id"] = StateMachine.timeout 
        self.LSL.EXO_stream_out(state_dict, trial_over = True)

    def set_trial_termination(self, state_dict):        
        state_dict["state_start_time"] = time()
        state_dict["state_wait_time"] = 0.5

    def set_exit_or_error(self, state_dict, background_color="green4", main_text="EXPERIMENT FINISHED", sub_text="Press ESC to exit or ENTER to restart."):
        state_dict["background_color"] = background_color
        state_dict["main_text"] = main_text 
        state_dict["sub_text"] = sub_text

    #### KEYBOARD HANDLERS
    @staticmethod
    def one_time_ENTER(state_dict):
        current_enter_state = pygame.key.get_pressed()[pygame.K_RETURN]
        if current_enter_state and not state_dict["previous_enter_state"]:
            state_dict["enter_pressed"] = True
        else:
            state_dict["enter_pressed"] = False
        state_dict["previous_enter_state"] = current_enter_state

    @staticmethod
    def one_time_ESCAPE(state_dict):
        current_escape_state = pygame.key.get_pressed()[pygame.K_ESCAPE]
        if current_escape_state and not state_dict["previous_escape_state"]:
            state_dict["escape_pressed"] = True
        else:
            state_dict["escape_pressed"] = False
        state_dict["previous_escape_state"] = current_escape_state

    @staticmethod
    def latch_SPACE(state_dict):
        current_space_state = pygame.key.get_pressed()[pygame.K_SPACE]
        if current_space_state and not state_dict["previous_space_state"]:
            state_dict["space_pressed"] = not state_dict.get("space_pressed", False)
        state_dict["previous_space_state"] = current_space_state

    #### TRIAL GENERATION
    @staticmethod
    def exo_condition_name(condition_code):
        conditions = {0: "resist", 1: "assist", 2: "transparent", 99: "None"}
        return conditions.get(int(condition_code), "None")

    @staticmethod
    def generate_familiarization_trials(n):
        """
        Generate familiarization trials with only incorrect executions (0).
        The familiarization trials are used to let the subject get used to the task.
        The events are split evenly between "UP" (1) and "DOWN" (0) trials.
        If n is odd, the extra trial is assigned as "UP".
        Returns:
            np.ndarray: Array of shape (n, 5) with columns [event, execution, torque_profile, torque, exo_condition].
        """
        ones = np.ones(n // 2, dtype=int)
        zeros = np.zeros(n // 2, dtype=int)
        if n % 2 != 0:
            ones = np.append(ones, 1)  # Ensure UP gets the extra trial
        events = np.append(ones, zeros)
        executions = torque_profiles = torques = exo_conditions = 99 * np.ones(n, dtype=int) # Placeholder, not used in familiarization
        trials = np.column_stack((events, executions, torque_profiles, torques, exo_conditions))
        np.random.shuffle(trials)
        return trials

    def generate_trials(self, state_dict: dict) -> np.ndarray:
        """
        Generates all experiment trials, including familiarization, main, and end control trials.
        Populates self.events, self.correctness_list, self.torque_profile_list, self.torque_magnitude_list,
        and self.exo_condition_list.
        Updates state_dict with total trial count and familiarization trial count.

        Args:
            state_dict (dict): Dictionary containing experiment configuration, including:
                - trial_conditions: dict of trial condition parameters
                - familiarization_trials_No: number of familiarization trials
                - end_control_trials: number of end control trials
                - exo_parameters: dict with torque_limit
                - randomize_trials: bool, whether to shuffle main trials
                - trial_randomization: randomization protocol name

        Returns:
            np.ndarray: Array of all trials, where each row represents a trial with columns:
                [event type, execution correctness, torque profile, torque magnitude, EXO condition].
        """

        def generate_trials_for_condition(condition_id, assistance, condition_trial_No, torque_profile, torque_magnitude):
            """
            Generates a set of trials for a given experimental condition.

            Args:
                condition_id: Identifier for the condition.
                assistance: "assist", "resist", or "transparent" (determines execution correctness and torque).
                condition_trial_No: Number of trials for this condition.
                torque_profile: Profile name or "random".
                torque_magnitude: Torque value for all trials.

            Returns:
                np.ndarray: Each row is [event type, execution correctness, torque profile, torque magnitude, EXO condition].
            """
            # Split events evenly between "UP" (1) and "DOWN" (0)
            ones = np.ones(condition_trial_No // 2, dtype=int)
            zeros = np.zeros(condition_trial_No // 2, dtype=int)
            events = np.append(ones, zeros)

            # If odd number of trials, add an extra "UP"
            if condition_trial_No % 2 != 0:
                events = np.append(events, int(1))  # Extra UP

            # Set EXO condition and execution correctness.
            if assistance == "assist":
                executions = np.ones(condition_trial_No)
                exo_conditions = np.ones(condition_trial_No)
            elif assistance == "resist":
                executions = np.zeros(condition_trial_No)
                exo_conditions = np.zeros(condition_trial_No)
            elif assistance == "transparent":
                executions = np.ones(condition_trial_No)
                exo_conditions = 2 * np.ones(condition_trial_No)
                torque_magnitude = 0
            else:
                self.logger.error(f"Invalid assistance type: {assistance} in condition {condition_id}.")
                return None  # Invalid assistance type

            # Assign torque profiles
            if torque_profile == "random":
                # Randomly assign a profile for each trial
                torque_profile_labels = []
                for i in range(condition_trial_No):
                    torque_profile_labels.append(random.choice(list(self.profiles_dict.values())))
            elif torque_profile in self.profiles_dict:
                # Use the specified profile for all trials
                torque_profile_labels = [self.profiles_dict[torque_profile]] * condition_trial_No
            else:
                self.logger.error(f"Invalid torque profile: {torque_profile} in condition {condition_id}.")
                return None  # Invalid torque profile

            # Clamp torque magnitude to exoskeleton limit if needed
            if torque_magnitude > state_dict["exo_parameters"]["torque_limit"]:
                self.logger.warning(
                    f"Torque magnitude ({torque_magnitude} Nm) exceeds limit, setting to {state_dict['exo_parameters']['torque_limit']} Nm."
                )
                torque_magnitude = state_dict["exo_parameters"]["torque_limit"]

            torque_magnitude_labels = [torque_magnitude] * condition_trial_No

            trial_rules_for_condition = np.column_stack((events, executions, torque_profile_labels, torque_magnitude_labels, exo_conditions))
            np.random.shuffle(trial_rules_for_condition)  # Shuffle trials within the condition
            return trial_rules_for_condition

        # --- Generate familiarization trials (always incorrect execution) ---
        familiarization_trials_rules = self.generate_familiarization_trials(state_dict["familiarization_trials_No"])
        state_dict["familiarization_trial_No"] = familiarization_trials_rules.shape[0]

        # --- Generate end control trials if needed (same as familiarization) ---
        if state_dict["end_control_trials"] != 0:
            end_control_trial_rules = self.generate_familiarization_trials(state_dict["end_control_trials"])

        # --- MAIN TRIALS (VARYING EXECUTION CORRECTNESS, TORQUE PROFILE AND TORQUE LEVEL) ---
        # For each condition, generate the corresponding trials
        condition_trials_by_exo_condition = {0: [], 1: [], 2: []}
        for condition_id, (assist, condition_No, torque_profile, torque) in state_dict["trial_conditions"].items():
            condition_trials = generate_trials_for_condition(condition_id, assist, condition_No, torque_profile, torque)
            if condition_trials is not None:
                exo_condition = int(condition_trials[0, 4])
                condition_trials_by_exo_condition[exo_condition].append(condition_trials)

        def stack_condition_trials(condition_trials):
            if condition_trials:
                return np.vstack(condition_trials)
            return np.empty((0, 5))

        def shuffle_if_needed(trials):
            if state_dict["randomize_trials"]:
                np.random.shuffle(trials)
            return trials

        # --- SHUFFLE MAIN TRIALS ---
        if state_dict["trial_randomization"] == "resist_then_assist_with_transparent":
            resist_trials = stack_condition_trials(condition_trials_by_exo_condition[0])
            assist_trials = stack_condition_trials(condition_trials_by_exo_condition[1])
            transparent_trials = stack_condition_trials(condition_trials_by_exo_condition[2])

            resist_block = shuffle_if_needed(np.vstack((resist_trials, transparent_trials)))
            assist_block = shuffle_if_needed(np.vstack((assist_trials, transparent_trials)))
            main_trial_rules = np.vstack((resist_block, assist_block))
        elif state_dict["trial_randomization"] == "assist_then_resist_with_transparent":
            resist_trials = stack_condition_trials(condition_trials_by_exo_condition[0])
            assist_trials = stack_condition_trials(condition_trials_by_exo_condition[1])
            transparent_trials = stack_condition_trials(condition_trials_by_exo_condition[2])

            assist_block = shuffle_if_needed(np.vstack((assist_trials, transparent_trials)))
            resist_block = shuffle_if_needed(np.vstack((resist_trials, transparent_trials)))
            main_trial_rules = np.vstack((assist_block, resist_block))
        else:
            all_condition_trials = []
            for condition_trials in condition_trials_by_exo_condition.values():
                all_condition_trials.extend(condition_trials)
            main_trial_rules = shuffle_if_needed(np.vstack(all_condition_trials))

        # --- COMBINE ALL TRIALS ---
        # Stack familiarization, main, and end control trials into one array
        if state_dict["end_control_trials"] != 0:
            final_trials = np.vstack((familiarization_trials_rules, main_trial_rules, end_control_trial_rules))
        else:
            final_trials = np.vstack((familiarization_trials_rules, main_trial_rules))

        # Update state_dict and internal lists for use in the state machine
        state_dict["trials_No"] = final_trials.shape[0]
        self.events = final_trials[:, 0]
        self.correctness_list = final_trials[:, 1]
        self.torque_profile_list = final_trials[:, 2]
        self.torque_magnitude_list = final_trials[:, 3]
        self.exo_condition_list = final_trials[:, 4]

        return final_trials
