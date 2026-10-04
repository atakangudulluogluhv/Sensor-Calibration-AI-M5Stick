# Sensor Calibration on an M5StickC Plus2

Embedded part of our Deep Learning (DIL700) term paper at University West.
The model comparison in Python is in
[Sensor-Calibration-AI-HV](https://github.com/atakangudulluogluhv/Sensor-Calibration-AI-HV).

A sensor that drifts reads y = f(x) instead of the true value x; calibration
means learning f⁻¹. Here the ESP32 does that on its own: it trains a linear
model with gradient descent on the device, then uses it to correct the
readings live on the screen.

## How it works

- `dataset.h` holds 3000 PM2.5 values from the
  [Air Pollution in Seoul](https://www.kaggle.com/datasets/bappekim/air-pollution-in-seoul)
  dataset, stored as a C array so no SD card or sensor is needed.
- The faulty sensor is simulated as `reading = 1.15 * true + 5`
  (15% gain error plus an offset).
- On boot the device runs 5000 epochs of SGD for `y = w*x + b` and shows
  the loss every 100 epochs. When it finishes it shows the training time and
  the learned `w` and `b`.
- After that it loops through the dataset showing the real, faulty and
  calibrated value for each sample.

No ML library is used; the training loop and inference are plain C++.

The simulated drift is linear, so a linear model can undo it exactly: the
ideal values are w = 1/1.15 ≈ 0.870 and b = −5/1.15 ≈ −4.35. After 5000
epochs the same loop run in Python ends at w ≈ 0.868, b ≈ −4.29, with an MSE
of about 0.001 against the true values.
The point is to show training and inference running on a microcontroller,
not to model a real sensor's drift.

## Photos

![Training](https://github.com/user-attachments/assets/c8ca23d7-b235-4503-a420-e0f7a58ebf77)
![Result](https://github.com/user-attachments/assets/c05838e7-57ff-4107-be69-88d3190024c8)
![Calibrated readings](https://github.com/user-attachments/assets/02dd14c1-3749-4162-88d5-c633f44fb3bb)

## Running

1. Install the `M5Unified` library in the Arduino IDE.
2. Open `sensor_calibration/sensor_calibration.ino`.
3. Select the M5StickC Plus2 board and upload.

Training progress is also printed on the serial port at 115200 baud.
