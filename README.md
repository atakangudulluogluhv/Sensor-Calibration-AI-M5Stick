# Edge AI Sensor Calibration on ESP32 (M5StickC Plus2)

## Overview
This repository demonstrates a complete **Edge AI / TinyML** pipeline for real-time sensor calibration using an ESP32 microcontroller (M5StickC Plus2). It addresses common hardware constraints—such as sensor drift, bias, and noise—by running Machine Learning algorithms directly on the edge device, eliminating the need for cloud computing or heavy external processing.

Using a PM2.5 air quality section from "Air Pollution on Seuol" dataset, the project simulates a faulty sensor and corrects it in real-time using mathematical inferences derived from a trained Machine Learning model. 

## Key Features
* **Hardware-in-the-Loop Simulation:** Runs a localized dataset directly on the ESP32 to simulate live sensor readings without needing the physical PM2.5 sensor.
* **Lightning-Fast Inference:** Executes calibration mathematics (`y = mx + b`) in less than a millisecond per reading.
* **On-Device Training (OLS & SGD):** Includes a standalone training script where the ESP32 calculates its own weights and biases using Ordinary Least Squares and Gradient Descent algorithms—no, Python backend required.
* **Lightweight C++ Implementation:** Bypasses heavy ML libraries (like TensorFlow Lite) for highly optimized, low-memory C++ execution.

##  Repository Structure
* `Sensor Calibration using AI.ipynb`: The primary data science notebook. Contains dataset analysis, simulated sensor degradation (15% drift + offset), and Scikit-Learn model evaluations (Linear Regression, Random Forest, SVR, RNN).
* `Measurement_summary_ilk_3000.csv`: A sample subset of the PM2.5 dataset used for the simulation.
* `dataset.h`: The C++ header file containing the 3000 data points, converted into a lightweight array for onboard ESP32 memory storage.
* `Edge_AI_Inference.ino`: The main Arduino sketch. It pulls raw data, applies the pre-trained weights, and displays the true vs. calibrated values on the M5StickC screen.
* `On_Device_Training.ino`: An advanced sketch where the ESP32 trains the linear regressor model entirely on its own processor.

## Results & Visualization
By extracting the weights from the Python-trained model and deploying them to the microcontroller, the ESP32 successfully suppresses the simulated noise and drift. 
![IMG_0670](https://github.com/user-attachments/assets/c8ca23d7-b235-4503-a420-e0f7a58ebf77)
![IMG_0674](https://github.com/user-attachments/assets/c05838e7-57ff-4107-be69-88d3190024c8)
![IMG_0678](https://github.com/user-attachments/assets/02dd14c1-3749-4162-88d5-c633f44fb3bb)

## Hardware & Software Requirements
* **Hardware:** M5StickC Plus2 (ESP32-PICO-V3-02) or any ESP32 development board.
* **Framework:** Arduino IDE (with `M5Unified` library installed).
* **Data Science Stack:** Python 3.x, Pandas, Numpy, Scikit-Learn, Matplotlib.

## How to Run
1. Clone this repository to your local machine.
2. Open the `Edge_AI_Inference.ino` file using Arduino IDE.
3. Ensure that `dataset.h` is located in the same directory as the `.ino` sketch.
4. Select the M5StickC Plus2 board and compile/upload the code.
5. Watch the real-time AI calibration on the device's display.
