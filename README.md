# Eduexo Experiment (PC)

This project integrates a Brain-Machine Interface (BMI) to train a neural network for classifying human intentions during experiments. Participants are asked to think about moving their arm up or down, and the EXO device either aids or contradicts their movement. EEG signals are recorded during these tasks, which are later used to classify the participant's intentions. This setup allows for precise control and monitoring of the experiment and communication with both EXO and computer which reads 'Events stream' in real time.

## Prerequisites

- Python 3.x
- Git
- Visual Studio Code (recommended)
- Compatible hardware for data streaming (e.g., EXO device)
    - The EXO device must meet specific requirements for compatibility. Ensure that your device firmware is up-to-date and that it supports the necessary data streaming protocols. Refer to: [https://github.com/JakaHatlakLJ/eduexo_EEG]

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Eduexo_PC
```

2. Install all requriements from `requirements.txt` in your environment:
```bash
pip install -r requirements.txt
```

## Project Hierarchy

The project directory is organized as follows. Once `experiment_do.py` is run for the first time with the default `experiment_config.json`, all subfolders will automatically be created and populated if not already.

```
Eduexo_PC/
├── main/                               # Main scripts
│   ├── experiment_do.py                # Run experiment
│   ├── experiment_interface.py         # Interface logic
│   ├── experiment_state_machine.py     # State machine
│   ├── experiment_logging.py           # Logging functions
│   ├── experiment_LSL.py               # LSL integration
│   └── experiment_config.json          # Configuration file
├── analysis/                           # Analysis scripts
│   ├── experiment_results/             # Results storage
│   ├── frequency_data/                 # Frequency data
│   └── jupyter/                        # Jupyter notebooks
│       ├── EXO_frequency_test/         # Control frequency analysis
│       ├── PC_torque_test              # Torque delivery analysis
│       └── analysis.ipynb              # ipynb notebook for analysis
├── testing_&_debugging/                # Testing scripts
│   ├── LSL_inlet.py                    # LSL inlet
│   ├── LSL_outlet.py                   # LSL outlet
│   ├── LSL_parameter_sender.py         # Send parameters to EXO
│   ├── LSL_predictions_inlet.py        # Test predictions inlet
│   ├── LSL_read_events_stream.py       # Test events stream
│   └── LSL_synthetic_predictions.py    # Test real event decoding
├── README.md                           # Documentation
├── requirements.txt                    # Dependencies
├── state_machine_diagram.drawio        # Documentation
└── state_machine_diagram.pdf           # Documentation
```

## Setting Up the Experiment

Before running the `experiment_do.py` script, you need to follow these steps:

1. **Configure and setup EXO:**
    - Install all requirements on EXO and make sure you can run scripts on it.
    - Refer to: [https://github.com/JakaHatlakLJ/eduexo_EEG]

2. **Configure the Experiment:**
    - Open `experiment_config.json`.
    - Modify the configuration parameters as needed for your experiment.

## Running the Experiment

1. Once the setup is complete, you can run the experiment using the following command:
```sh
python experiment_do.py
```

2. Run `EXO_main.py` on EXO at the same time and follow the instructions. (If run for testing purposes you can additionaly use LSL_send_position.py on EXO)

3. If run with real decoding, another LSL stream of name "PredictionStream" is needed. For testing run the following script in a seperate terminal to mimic the real decoder:
```sh
python /testing_&_debugging/LSL_synthetic_predictions.py
```

This will start the experiment based on the configurations prepared in the previous steps.

## Additional Information

- Refer to the docstrings and comments within each script for more detailed instructions and explanations.
- The `requirements.txt` file should contain all the necessary Python packages needed to run the project.

### `experiment_config.json` explanation

```json
{
    "experiment": {
        "real_time_classifier_prediction": 1            "Flag to choose experimet with real time EEG classifier decoding (1) or synthetic events experiment (0)",
        "define_trial_states": ["wait", "intend"]       "List of states during trial, combination of: 'wait'; 'imagine'; 'intend', or empty list if None",
        "start_time_range" : [1, 1.5]                   "Range of times in WAIT state.",
        "state_wait_time_range": [1, 1]                 "Range of times in WAIT state.",
        "imagination_time_range": [5, 5]                "Range of times for IMAGINATION phase.",
        "intention_time_range": [2, 3]                  "Range of times for INTENTION phase.",
        "trial_timeout": 3                              "Timeout duration for each trial.",
        "number_of_familiarization_trials": 2           "Number of familiarization trials at the begining (without EXO active)",
        "number_of_end_control_trials": 2               "Number of control trials at the end (without EXO active).",
        "trial_conditions": {
            "1": ["assist", 6, "sinusoidal", 1]         "Defines trial conditions as lists: [assistance type, number of trials, torque profile, torque magnitude]",
            "2": ["resist", 10, "sinusoidal", 3]        "Assistance type: either 'assist', 'resist', or 'transparent'",
            "3": ["assist", 14, "rectangular", 0.2]     "number of trials: (int) split into UP and DOWN trials, if uneven extra UP trial is added",
            "4": ["resist", 8, "random", 1.7]           "torque profile: 'sinusoidal', 'rectangular', 'triangular', 'trapezoid', 'smooth_trapezoid' or 'random' for random profile inside condition"
        }                                               "torque magnitude: maximum torque in Nm during trial, if larger than 'torque_limit', it is set to 'torque_limit'",
        "randomize_trials": 1                           "Flag to randomize trials inside the selected protocol (1) or leave them in condition groups (0)",
        "trial_randomization": "full_random"            "Trial randomization protocol. Use 'full_random' to shuffle all main trials together or 'resist_then_assist_with_transparent' to run randomized resist+transparent trials followed by randomized assist+transparent trials"
    },
    "exo_parameters":{
        "forearm_attachment_leverage_mm": 180           "Distance from load cell to the rotation axis",
        "maximum_arm_position_deg": 165                 "Maximum arm position in degrees. (straight arm is 180 deg)",
        "minimum_arm_position_deg": 55                  "Minimum arm position in degrees.",
        "center_offset_deg": 3                          "Defines a ± offset from the central arm position; if the arm is within this range, it is considered to be in the middle.",
        "edge_offset_deg": 5                            "Defines an offset from the maximum/minimum position at which the torque from exo is turned off",
        "torque_limit": 8                               "Torque limit on the EXO, if surpassed EXO returns ERROR and Torque is turned off",  
        "incorect_execution_time_control": 0            "Flag for choosing incorrect execution mode, (1) for time dependant pertrubation, (0) for position dependant pertrubation",
        "incorrect_execution_time_ms": 1500             "Time duration of time dependant pertrubation in ms",
        "PID_control": 0                                "Flag for enabling PID control during resist executions",
        "PID_parameters": {
            "FKp": 0.1                                  "Proportional Gain of force measurement", 
            "FKd": 0.035                                "Derivative Gain of force measurement",
            "VKp": 0.015                                "Proportional gain of velocity measurement"
        }

    },
    "interface_data": {
        "full_screen_mode": 0                           "Flag for choosing full screen mode",
        "data_stream_interval": 0.01                    "Interval for motor parameters streaming.", 
        "save_data": 1                                  "Flag to save data (1 to save, 0 not to save).",
        "results_path":"./analysis/experiment_results"  "Path to save experiment results.",
    },
    "participant": {
        "age": 22                                       "Age of the participant.",
        "id": 1                                         "ID of the participant.",
        "name":"Jaka"                                   "Name of the participant."
    }
}


{
    "experiment": {
        "real_time_classifier_prediction": 0,
        "define_trial_states": [],
        "start_time_range" : [1, 1.5],
        "state_wait_time_range": [1, 1],
        "imagination_time_range": [3, 3],
        "intention_time_range": [1, 1],
        "trial_timeout": 10,
        "number_of_familiarization_trials": 0,
        "number_of_end_control_trials": 0,
        "trial_conditions": {
            "1": ["resist", 2, "sinusoidal", 2],
            "2": ["resist", 2, "trapezoid", 3],
            "3": ["assist", 2, "sinusoidal", 2],
            "4": ["assist", 2, "trapezoid", 3]
        },
        "randomize_trials": 1
    },
    "exo_parameters":{
        "forearm_attachment_leverage_mm": 180,
        "maximum_arm_position_deg": 165,
        "minimum_arm_position_deg": 55,
        "center_offset_deg": 3,
        "edge_offset_deg": 5,
        "torque_limit": 8,
        "incorect_execution_time_control": 0,
        "incorrect_execution_time_ms": 1500,
        "PID_control": 0,
        "PID_parameters": {
            "FKp": 0.1,
            "FKd": 0.035,
            "VKp": 0.015
        }
    },
    "interface_data": {
        "full_screen_mode": 0,
        "data_stream_interval": 0.01,
        "save_data": 1,
        "results_path": "./analysis/experiment_results"  
    },
    "participant": {
        "age": 22,
        "id": 1,
        "name": "Jaka"
    }
}

```
### Additional Scripts:

   You can run other scripts for specific functionalities or testing as needed. For example, to test if "events_stream" is sending out correct data, use this in a seperate terminal:
   ```sh
   python LSL_read_events_stream.py
   ```
