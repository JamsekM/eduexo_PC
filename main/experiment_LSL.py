from pylsl import StreamInfo, StreamOutlet, local_clock, resolve_byprop, StreamInlet
from time import perf_counter
import threading
import json
import logging

class LSLHandler:
    """
    Handles all Lab Streaming Layer (LSL) communication for the Eduexo experiment.
    Responsible for sending setup/instructions/events to the exoskeleton and classifier,
    and for receiving data and predictions from the exoskeleton and classifier.
    """

    def __init__(self, state_dict: dict, receive: bool=True, send: bool=True, predict: bool=False):
        """
        Initialize the LSLHandler class and set up all required LSL streams.

        :param state_dict: Dictionary containing the current state information.
        :param receive: Flag to enable receiving data from LSL stream.
        :param send: Flag to enable sending data to LSL stream.
        :param predict: Flag to enable receiving predictions from LSL stream.
        """
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("LSL")
        self.logger = logger
        self.logger_predictions = logging.getLogger("Predictions")
        self.timestamp_g = local_clock()
        self.missed_samples = 0
        self.previous_time = perf_counter()

        if send:
            # Create LSL stream for sending SET UP instructions to EXO
            info_SETUP_EXO = StreamInfo(
                'EXO_SETUP',            # name
                'SETUP',                # type
                1,                      # channel_count
                0,                      # nominal rate=0 for irregular streams
                'string',               # channel format
                'Eduexo_PC'             # source_id
            )
            self.outlet_SETUP_EXO = StreamOutlet(info_SETUP_EXO)
            logger.info("Stream to Setup EXO is online...")

            # Create LSL stream for sending instructions to EXO (main control)
            info_EXO = StreamInfo(
                'EXOInstructions',      # name
                'Instructions',         # type
                4,                      # channel_count
                0,                      # nominal rate=0 for irregular streams
                'float32',              # channel format
                'Eduexo_PC'             # source_id
            )
            self.outlet_EXO = StreamOutlet(info_EXO)
            logger.info("Stream to EXO is online...")

            # Create LSL stream for sending discrete events to classifier
            info_events = StreamInfo(
                'ExperimentEvents',     # name
                'Events',               # type
                1,                      # channel_count
                0,                      # nominal rate=0 for irregular streams
                'string',               # channel format
                'Eduexo_PC1'            # source_id
            )

            # Add channel metadata
            desc_events = info_events.desc()
            channels_events = desc_events.append_child("channels")
            channel = channels_events.append_child("channel")
            channel.append_child_value("label", "event_marker")
            channel.append_child_value("type", "trigger")
            channel.append_child_value("unit", "n/a")

            self.outlet_events = StreamOutlet(info_events)
            logger.info("Stream for events to classifier is online...")

            # Create LSL stream for sending continuous motor data to classifier
            info_events_continuous = StreamInfo(
                'ExoEvents',            # name
                'EventsContinuous',     # type
                3,                      # channel_count
                100,                    # nominal rate=0 for irregular streams
                'float32',              # channel format
                'Eduexo_PC2'            # source_id
            )

            # Add channel metadata
            desc_continuous = info_events_continuous.desc()
            channels_cont = desc_continuous.append_child("channels")

            # Channel 1: Position
            ch1 = channels_cont.append_child("channel")
            ch1.append_child_value("label", "position")
            ch1.append_child_value("type", "angle")
            ch1.append_child_value("unit", "deg")

            # Channel 2: Velocity
            ch2 = channels_cont.append_child("channel")
            ch2.append_child_value("label", "velocity")
            ch2.append_child_value("type", "angular_velocity")
            ch2.append_child_value("unit", "deg/s")

            # Channel 3: Torque
            ch3 = channels_cont.append_child("channel")
            ch3.append_child_value("label", "torque")
            ch3.append_child_value("type", "force")
            ch3.append_child_value("unit", "Nm")

            self.outlet_events_continuous = StreamOutlet(info_events_continuous)
            logger.info("Stream for motor data to classifier is online...")
        
        if receive:
            # Resolve LSL stream for receiving EXO data and create an inlet
            logger.info("Looking for LSL stream of type: 'EXO'...")
            while True:
                streams = resolve_byprop('type', 'EXO', timeout=5)
                if streams:
                    break
                logger.warning("No LSL stream found of type: 'EXO'. Retrying...")
            self.inlet = StreamInlet(streams[0])
            logger.info("Receiving data from EXO...")

        if predict:
            # Resolve "PredictionStream" for receiving decoded events
            logger.info("Looking for LSL stream of name: 'PredictionStream'...")
            while True:
                streams_p = resolve_byprop('name', 'PredictionStream', timeout=5)
                if streams_p:
                    break
                logger.warning("No LSL stream found of name: 'PredictionStream'. Retrying...")
            self.predictions_inlet = StreamInlet(streams_p[0])
            logger.info("Receiving data from EXO...")

        # Send initial setup data to EXO
        self.send_setup_data(state_dict["exo_parameters"])
        
    def send_setup_data(self, exo_config: dict):
        """
        Send setup data to EXO using LSL.
        :param exo_config: Dictionary containing the EXO configuration parameters.
        """

        setup_EXO_data = json.dumps(exo_config)
        self.outlet_SETUP_EXO.push_sample([setup_EXO_data])
        self.logger.info("Setup data sent to EXO.")

    def stream_events_data(self, stop_event: threading.Event, state_dict: dict, the_lock: threading.Lock):
        """
        Continuously stream position/torque data (at a fixed interval)
        and send an 'event' every time a new event occurs using LSL.

        :param stop_event: Event to signal stopping the streaming.
        :param state_dict: Dictionary containing the current state information.
        :param the_lock: Lock to synchronize access to shared resources.
        """
        # Configuration
        data_interval = state_dict["data_stream_interval"]  # how often we send 'data' samples
        last_data_time = perf_counter()
        old_event = 99
        self.timestamp_g = local_clock()

        while not stop_event.is_set():
            if state_dict["stream_online"]:
                try:
                    # 1) Stream position/torque (data) at regular intervals
                    current_time = perf_counter()
                    self.timestamp = local_clock()
                    with the_lock:
                        self.timestamp_g = self.timestamp
                    if current_time - last_data_time >= data_interval:
                        position = state_dict["current_position"]   # current position
                        velocity = state_dict["current_velocity"]   # current velocity
                        torque = state_dict["current_torque"]       # current torque
                        sample = [position, velocity, torque]
                        self.outlet_events_continuous.push_sample(sample, timestamp=self.timestamp)
                        # self.logger.info(sample)

                        last_data_time = current_time

                    # 2) Send an event once every time a new event happens
                    if old_event != state_dict["event_id"] and not state_dict["event_id"] == 99:
                        event_id = state_dict["event_id"]
                        event_type = state_dict["event_type"]
                        torque_profile = state_dict["torque_profile"]
                        torque_magnitude = state_dict["torque_magnitude"]
                        exo_condition = state_dict["exo_condition"]
                        event_data = {
                            'Sample_Type': 'event',
                            'Event_ID': event_id,
                            'Event_Type': event_type,
                            'TorqueProfile': torque_profile,
                            'TorqueMagnitude': torque_magnitude,
                            'ExoCondition': exo_condition,
                            'Event_Timestamp': self.timestamp
                        }
                        event_json_str = json.dumps(event_data)
                        self.outlet_events.push_sample([event_json_str], timestamp=self.timestamp)
                        self.logger.info(event_json_str)
                        old_event = state_dict["event_id"]

                except Exception as e:
                    self.logger.error(f"Error in streaming Events data: {e}")

        self.logger.info("Stopped streaming Events data.")

    def EXO_stream_in(self, state_dict: dict):
        """
        Receive data from EXO and update the state dictionary.

        :param state_dict: Dictionary containing the current state information.
        """
        # Receive data from EXO
        current_time = perf_counter()
        self.inlet.flush()    
        sample, timestamp = self.inlet.pull_sample(timeout=0.1)

        if sample is None:
            self.missed_samples += 1
            # Consider stream offline if we miss N consecutive samples
            if self.missed_samples >= 50:
                if current_time - self.previous_time >= 3:
                    self.logger.error("Stream lost! Trying to reconnect...")
                    state_dict["current_position"] = None
                    state_dict["stream_online"] = False
                    self.previous_time = current_time
                    self.send_setup_data(state_dict["exo_parameters"])
        else:
            self.missed_samples = 0  # Reset counter if we got a sample
            state_dict["stream_online"] = True
            state_dict["current_position"] = round(sample[0], 5)
            state_dict["current_velocity"] = round(sample[1], 5)
            state_dict["current_torque"] = round(sample[2], 5)
            state_dict["exo_execution"] = sample[3]
            state_dict["desired_torque"] = round(sample[4], 5)
            state_dict["demanded_torque"] = round(sample[5], 5)
            state_dict["measured_torque"] = round(sample[6], 5)

    def EXO_stream_out(self, state_dict: dict = None, torque_profile: int = 1, torque_magnitude: float = 1, correctness: int = 1, trial_over = False, experiment_over = False):
        """
        Send a TorqueProfile, direction, and Correctness once every time a new event happens.
        
        :param state_dict: Dictionary containing the current state information.
        :param torque_profile: Torque profile to be sent (int, see t_profile_dict).
        :param torque_magnitude: Magnitude of the torque to be sent.
        :param correctness: Correctness of the execution (1=correct, 0=incorrect).
        :param trial_over: Flag indicating if the trial is over (sends stop signal).
        :param experiment_over: Flag indicating if the experiment is over (sends termination signal).
        """
        # Determine direction and correctness based on state and prediction
        if experiment_over:
            direction = 99  # Special code for experiment termination
        elif trial_over:
            direction =  0  # Special code for trial end
            torque_magnitude = 0
        else:
            if state_dict["real_time_prediction"]:
                # Use prediction to determine correctness and direction
                if state_dict["trial"] == "UP":
                    direction = 10
                    if state_dict["prediction"] == "UP":
                        correctness = 1
                    else:
                        correctness = 0
                else:
                    direction = 20
                    if state_dict["prediction"] == "UP":
                        correctness = 0
                    else:
                        correctness = 1
            else:
                # Determine direction based on trial type and correctness
                if state_dict["trial"] == "UP":
                    direction = 10
                else:
                    direction = 20

        # Send instructions data to EXO
        instructions_data = [int(torque_profile), int(correctness), int(direction), float(torque_magnitude)]
        self.outlet_EXO.push_sample(instructions_data)

        # Update state_dict with human-readable info if not ending trial/experiment
        if not experiment_over and not trial_over:
            # Map torque profile index to its corresponding name for display/logging
            t_profile_dict = {0: "trapezoid", 1: "triangular", 2: "sinusoide", 3: "rectangular", 4: "smoothed_trapezoid"}
            state_dict["torque_profile"] = t_profile_dict[torque_profile]
            state_dict["torque_magnitude"] = float(torque_magnitude)
            state_dict["correctness"] = correctness

    def get_predictions(self, stop_event: threading.Event, state_dict: dict = None, verbose: bool=False):
        """
        Continuously receive predictions from the EXO and update the state dictionary.

        :param stop_event: Event to signal stopping the prediction receiving.
        :param state_dict: Dictionary containing the current state information.
        :param verbose: Flag to enable verbose logging of predictions.
        """
        if state_dict is None:
            state_dict = {}
        previous_time = perf_counter()
        while not stop_event.is_set():
            self.predictions_inlet.flush()
            recieved = False
            # Only receive predictions during relevant experiment states
            while state_dict["current_state"] in {"IMAGINATION", "INTENTION", "TRIAL_UP", "TRIAL_DOWN", "MOVING_UP", "MOVING_DOWN"} and not stop_event.is_set():
                current_time = perf_counter()
                # Limit prediction polling rate to 200 Hz
                if current_time - previous_time >= 1/200:
                    sample, timestamp = self.predictions_inlet.pull_sample(timeout=0.1)
                    if sample is not None and len(sample) > 0:
                        prediction_data = json.loads(sample[0])  # Parse JSON string from LSL
                        if not recieved:
                            if state_dict["activate_EXO"]:
                                # Update state_dict with the predicted event name (e.g., "UP" or "DOWN")
                                state_dict["prediction"] = prediction_data["predicted_event_name"]
                            recieved = True
                            if verbose:
                                self.logger_predictions.info(prediction_data)
                    previous_time = current_time
