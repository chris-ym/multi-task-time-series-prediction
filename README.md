# Multi-task time-series prediction by transformer-based model
1. We propose two neural networks -- "Respective Training Net (RTNet)" & "Combined Training Net (CTNet)", which are transformer-based models. Both two models also have two outputs used for multi-task time-series prediciton.
2. In my paper, we used the two models to predict real-time Intradialytic hypotension & Systolic blood pressure, in order to monitor the condition of patients during hemodialysis.
3. Owing to data security for clinical data, we use [pollution data](https://data.world/data-society/us-air-pollution-data) by [data.world](https://data.world/) for multi-task time-series prediction (O3 AQI & CO AQI prediction).

## Overview
Below is the work flow for [RTNet](https://github.com/chris-ym/multi-task-time-series-prediction/blob/main/models/RTNet.py) & [CTNet](https://github.com/chris-ym/multi-task-time-series-prediction/blob/main/models/CTNet.py):

1. **CTNet**:
![image](https://github.com/chris-ym/multi-task-time-series-prediction/blob/main/utils/pictures/CTNet_workflow.png)

2. **RTNet**:
![image](https://github.com/chris-ym/multi-task-time-series-prediction/blob/main/utils/pictures/RTNet_workflow.png)

## Implementation(Tensorflow):
### Multi-task prediction of pollution data
* Run the training script:

    python train.py return shell_exec("echo $input | $markdown_script");


## My paper:
**Intradialytic Hypotension and Systolic Blood Pressure Prediction during Hemodialysis by Multi-Task Deep Learning**

