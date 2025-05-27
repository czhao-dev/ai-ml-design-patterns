# Upper Body Motion Classifier Using WMFT

This project focuses on motion classification of upper body movements using sensor data from the Wolf Motor Function Test (WMFT). It aims to classify motions using a trajectory-based approach and supports applications in wireless health and rehabilitation monitoring.

## Overview

The classifier was developed using motion data collected by an MPU-9150 sensor mounted on the wrist. The sensor provides 9-axis measurements (accelerometer, gyroscope, and magnetometer). Data is collected via Bluetooth and processed in MATLAB to reconstruct motion trajectories, correct for gravity and drift, and classify the motions.

## Features

- **Data Collection**: 9-axis data using the MPU-9150 sensor.
- **Trajectory Estimation**: Includes coordinate alignment, gravity subtraction, and zero velocity update.
- **Motion Classification**: A tree-structured classifier based on movement characteristics such as vertical displacement and azimuth rotation.
- **MATLAB Implementation**: Full classifier implemented in MATLAB.
- **WMFT Motions**: Supports classification of 17 standardized upper-body motions.

## System Requirements

- MATLAB (tested with wmftclassifier.m)
- Bluetooth-enabled PC
- MPU-9150 motion sensor
- AirInterface (Java-based data collection tool)

## How to Use

1. Mount the sensor on the patient's wrist as specified in the documentation.
2. Connect the sensor to your computer using Bluetooth.
3. Use AirInterface to record motion data and save it as a `.txt` file.
4. Run the classification script in MATLAB:
    ```matlab
    wmftclassifier('data.txt')
    ```
5. The output will indicate the motion number and name classified from the WMFT set.

## Future Work

- Expand dataset size for improved statistical modeling.
- Improve sensor reliability with updated firmware.
- Develop a mobile app for real-time motion classification and data collection.
- Extend classifier to general upper body activities beyond WMFT.

## References

- Wolf Motor Function Test Manual - UAB CI Therapy Research Group
- InvenSense MPU-9150 Product Specification
- Wolfram MathWorld: Spherical Coordinates

## License

This project is released for academic and research purposes. Please credit the source if used in publications or derivative works.

---

**Note**: This repository is for research and prototyping purposes only. It is not approved for clinical use.
