// Trains a linear calibration model (y = w*x + b) on the M5StickC Plus2 with
// gradient descent, then runs it on the simulated faulty sensor readings.
//
// The faulty sensor is simulated as 15% gain error + 5 unit offset on top of
// the real PM2.5 values in dataset.h.

#include <M5Unified.h>
#include "dataset.h"

const float DRIFT_GAIN   = 1.15;
const float DRIFT_OFFSET = 5.0;

float w = 0.0;
float b = 0.0;

const float learning_rate = 0.0000005;
const int   epochs        = 5000;

int  sample_index  = 0;
bool training_done = false;

float faulty_reading(float true_value) {
    return true_value * DRIFT_GAIN + DRIFT_OFFSET;
}

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);

    M5.Display.setRotation(1);
    M5.Display.setTextSize(2);
    M5.Display.fillScreen(TFT_BLACK);
    Serial.begin(115200);

    M5.Display.setTextColor(TFT_YELLOW);
    M5.Display.setCursor(0, 10);
    M5.Display.println("Training...");

    unsigned long start_ms = millis();

    for (int epoch = 1; epoch <= epochs; epoch++) {
        float total_loss = 0.0;

        for (int i = 0; i < dataset_size; i++) {
            float y_true = dataset_pm25[i];
            float x      = faulty_reading(y_true);
            float y_pred = w * x + b;

            float err = y_true - y_pred;
            total_loss += err * err;

            // SGD step on squared error. The bias gets a larger step, otherwise
            // it barely moves at this learning rate.
            w += learning_rate * err * x;
            b += learning_rate * err * 10;
        }

        if (epoch % 100 == 0) {
            float mse = total_loss / dataset_size;
            M5.Display.fillRect(0, 60, 240, 70, TFT_BLACK);
            M5.Display.setCursor(0, 60);
            M5.Display.setTextColor(TFT_CYAN);
            M5.Display.printf("Epoch: %d/%d\n", epoch, epochs);
            M5.Display.printf("Loss: %.1f\n", mse);
            Serial.printf("Epoch: %d | Loss: %f | w: %f | b: %f\n", epoch, mse, w, b);
        }
    }

    unsigned long training_ms = millis() - start_ms;

    M5.Display.clear();
    M5.Display.setCursor(0, 0);
    M5.Display.setTextColor(TFT_GREEN);
    M5.Display.println("Finished!");

    M5.Display.setTextSize(1);
    M5.Display.setTextColor(TFT_WHITE);
    M5.Display.printf("Time: %lu ms\n", training_ms);
    M5.Display.printf("Weight(w): %f\n", w);
    M5.Display.printf("Bias(b): %f\n", b);

    delay(15000);   // time to read the result
    M5.Display.setTextSize(2);
    training_done = true;
}

void loop() {
    M5.update();
    if (!training_done) return;

    float true_value = dataset_pm25[sample_index];
    float faulty     = faulty_reading(true_value);
    float calibrated = faulty * w + b;

    M5.Display.clear();
    M5.Display.setCursor(0, 10);
    M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Display.printf("Real:   %.1f\n", true_value);

    M5.Display.setCursor(0, 50);
    M5.Display.setTextColor(TFT_RED, TFT_BLACK);
    M5.Display.printf("Faulty: %.1f\n", faulty);

    M5.Display.setCursor(0, 90);
    M5.Display.setTextColor(TFT_GREEN, TFT_BLACK);
    M5.Display.printf("Pred:   %.1f\n", calibrated);

    sample_index = (sample_index + 1) % dataset_size;
}
