# task_classification.py

import time
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_experiment(m, n, iterations):
    """
    Belirli bir m (örnek sayısı), n (boyut) ve iterasyon sayısı için
    deneyi 10 kez tekrarlar ve ortalamaları döndürür.
    """
    training_times = []
    errors = []

    for i in range(10): # Her senaryo 10 kez tekrarlanacak [cite: 71]
        # 1. Rastgele n-boyutlu veri seti üret (Binary Classification) [cite: 61]
        # n_informative=n yaparak tüm özelliklerin önemli olmasını sağlıyoruz.
        X, y = make_classification(n_samples=m, n_features=n, n_redundant=0, 
                                   n_informative=n, n_classes=2, random_state=None)

        # 2. Veriyi %70 eğitim, %30 test olarak ayır [cite: 62]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

        # 3. Single-layer Perceptron uygula [cite: 63]
        # max_iter: Tablo 1'deki iterasyon sayısı
        # eta0: Learning rate (varsayılan 1.0)
        clf = Perceptron(max_iter=iterations, tol=1e-3, random_state=None)

        # Eğitim süresini ölç
        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()

        # Süreyi milisaniye (ms) cinsinden kaydet
        training_time_ms = (end_time - start_time) * 1000
        training_times.append(training_time_ms)

        # Hatayı hesapla (Error = 1 - Accuracy)
        # Ödevde "Error (cost)" denmiş, genelde sınıflandırma hatası kastedilir.
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        error = 1.0 - accuracy
        errors.append(error)

    # 10 çalışmanın ortalamasını al [cite: 70]
    avg_time = np.mean(training_times)
    avg_error = np.mean(errors)

    return avg_time, avg_error

def main():
    print(f"{'Senaryo':<10} {'Iterasyon':<10} {'Tuple (m)':<12} {'Boyut (n)':<10} | {'Ort. Süre (ms)':<15} {'Ort. Hata':<10}")
    print("-" * 80)

    # Tablo 1'deki senaryolar [cite: 69]
    # Format: (Iterasyon, m, n)
    scenarios = [
        # 100 İterasyonluk durumlar
        (100, 10000, 100),   # a
        (100, 1000, 100),    # b (Tabloda sıralama karışık, m=1000 olan b şıkkı)
        (100, 100000, 100),  # c
        (100, 250000, 100),  # d
        
        # 500 İterasyonluk durumlar
        (500, 10000, 100),   # e
        (500, 1000, 100),    # f
        (500, 100000, 100),  # g
        (500, 250000, 100),  # h
    ]
    
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    for idx, (iters, m, n) in enumerate(scenarios):
        label = labels[idx]
        print(f"Çalıştırılıyor: Senaryo {label} (m={m}, n={n}, iter={iters})...")
        
        avg_time, avg_error = run_experiment(m, n, iters)
        
        # Sonuçları yazdır
        print(f"SONUÇ {label}:   {iters:<10} {m:<12} {n:<10} | {avg_time:.2f} ms        {avg_error:.4f}")
        print("-" * 80)

if __name__ == "__main__":
    main()