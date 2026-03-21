#include <M5Unified.h>
#include "dataset.h" // 3000 satırlık verimiz


float w = 0.0; // Starting weights.
float b = 0.0; // Starting Bias.


float learning_rate = 0.0000005; 
int epochs = 5000; 

int gecerli_index = 0;
bool egitim_tamamlandi = false;

void setup() {
    auto cfg = M5.config();
    M5.begin(cfg);
    
    M5.Display.setRotation(1);
    M5.Display.setTextSize(2);
    M5.Display.fillScreen(TFT_BLACK);
    Serial.begin(115200);

    M5.Display.setTextColor(TFT_YELLOW);
    M5.Display.setCursor(0, 10);
   
    M5.Display.println("Starting...");

    unsigned long baslangic_zamani = millis();

    for (int epoch = 1; epoch <= epochs; epoch++) {
        float total_loss = 0.0;

        for (int i = 0; i < dataset_boyutu; i++) {
            float gercek_y = dataset_pm25[i]; 
            float hatali_x = (gercek_y * 1.15) + 5.0;

            // Forward Pass
            float tahmin_y = (w * hatali_x) + b;

            // Loss
            float hata = gercek_y - tahmin_y;
            total_loss += (hata * hata); // MSE (Mean Squared Error) temeli

            // Backpropagation
            w += learning_rate * hata * hatali_x;
            b += learning_rate * hata * 10; // Bias'ın güncellenmesine destek
        }
        if (epoch % 100 == 0) {
            M5.Display.fillRect(0, 60, 240, 70, TFT_BLACK); // Sadece yazı alanını temizle
            M5.Display.setCursor(0, 60);
            M5.Display.setTextColor(TFT_CYAN);
            M5.Display.printf("Epoch: %d/%d\n", epoch, epochs);
            M5.Display.printf("Loss: %.1f\n", total_loss / dataset_boyutu);
            Serial.printf("Epoch: %d | Loss: %f | w: %f | b: %f\n", epoch, total_loss / dataset_boyutu, w, b);
        }
    }

    unsigned long egitim_suresi = millis() - baslangic_zamani;

    // Final report for training
    M5.Display.clear();
    M5.Display.setCursor(0, 0);
    M5.Display.setTextColor(TFT_GREEN);
    M5.Display.println("Finished!");
    
    M5.Display.setTextSize(1);
    M5.Display.setTextColor(TFT_WHITE);
    M5.Display.printf("Time: %lu ms\n", egitim_suresi);
    M5.Display.printf("Weight(w): %f\n", w);
    M5.Display.printf("Bias(b): %f\n", b);
    
    delay(15000); // Given time to read report.
    M5.Display.setTextSize(2);
    egitim_tamamlandi = true;
}

void loop() {
    M5.update();
    if (!egitim_tamamlandi) return;
    
    // INFERENCE
    float gercek_pm25 = dataset_pm25[gecerli_index];
    float hatali_okuma = (gercek_pm25 * 1.15) + 5.0;
    float yapay_zeka_sonucu = (hatali_okuma * w) + b;

    M5.Display.clear();
    M5.Display.setCursor(0, 10);
    M5.Display.setTextColor(TFT_WHITE, TFT_BLACK);
    M5.Display.printf("Real Val: %.1f\n", gercek_pm25);
    
    M5.Display.setCursor(0, 50);
    M5.Display.setTextColor(TFT_RED, TFT_BLACK);
    M5.Display.printf("False Val: %.1f\n", hatali_okuma);
    
    M5.Display.setCursor(0, 90);
    M5.Display.setTextColor(TFT_GREEN, TFT_BLACK);
    M5.Display.printf("Pred: %.1f\n", yapay_zeka_sonucu);
    
    gecerli_index++;
    if (gecerli_index >= dataset_boyutu) {
        gecerli_index = 0;
    }
    
    delay(0); 
}
