# task_mlp_noise.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import random

def add_noise(X_data, noise_rate_percent):
    """
    Verilen veri setinin (X_data) belirli bir yüzdesine (%P) gürültü ekler.
    Gürültü kuralı: Rastgele 10 özellik seçilir ve x_new = |x - 16| yapılır.
    """
    if noise_rate_percent == 0:
        return X_data.copy()

    X_noisy = X_data.copy()
    n_samples, n_features = X_noisy.shape
    
    # Gürültü eklenecek satır sayısı
    n_noisy_samples = int(n_samples * (noise_rate_percent / 100))
    
    # Rastgele satır indeksleri seç
    noisy_indices = np.random.choice(n_samples, n_noisy_samples, replace=False)
    
    for idx in noisy_indices:
        # Rastgele 10 özellik (dimension) seç
        dims = np.random.choice(n_features, 10, replace=False)
        for d in dims:
            # Formül: |x - 16| (Pikseli tersine çevirme gibi)
            X_noisy[idx, d] = abs(X_noisy[idx, d] - 16)
            
    return X_noisy

def main():
    # 1. Veri Yükle
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # 2. %70 Train, %30 Test
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Tablo 2 Senaryoları
    # (Train Noise %, Test Noise %)
    scenarios = [
        ('a', 0, 25),
        ('b', 0, 50),
        ('c', 0, 75),
        ('d', 25, 0),
        ('e', 50, 0),
        ('f', 75, 0),
        ('g', 25, 25),
        ('h', 50, 50),
        ('i', 75, 75)
    ]

    print(f"{'Senaryo':<10} {'Train Noise %':<15} {'Test Noise %':<15} {'Test Accuracy':<15} {'Error (1-Acc)':<15}")
    print("-" * 75)

    # Standart MLP modeli
    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)

    # Senaryoları Çalıştır
    for label, train_n, test_n in scenarios:
        # Gürültü ekle
        X_train_noisy = add_noise(X_train_orig, train_n)
        X_test_noisy = add_noise(X_test_orig, test_n)
        
        # Ölçeklendirme (Scaling) - MLP için önemli
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_noisy)
        X_test_scaled = scaler.transform(X_test_noisy)
        
        # Eğit ve Test Et
        mlp.fit(X_train_scaled, y_train)
        acc = mlp.score(X_test_scaled, y_test)
        error = 1 - acc
        
        print(f"{label:<10} {train_n:<15} {test_n:<15} {acc:.4f}          {error:.4f}")

    # --- j. Denoising (Gürültü Temizleme) ---
    print("-" * 75)
    print("j. Denoising Strategy (PCA) applying on Scenario h (50% noise)...")
    
    # h senaryosunun verisini al (50% Train Noise, 50% Test Noise)
    X_train_h = add_noise(X_train_orig, 50)
    X_test_h = add_noise(X_test_orig, 50)
    
    # PCA ile Denoising Stratejisi
    # Mantık: Gürültü genellikle rastgele varyasyonlardır. PCA ile veriyi daha az boyuta indirip
    # tekrar geri oluşturursak (inverse_transform), ana yapı korunurken gürültü kaybolabilir.
    # 64 boyuttan 20 boyuta indiriyoruz.
    pca = PCA(n_components=20, random_state=42)
    
    X_train_denoised = pca.fit_transform(X_train_h) # Boyut indir
    X_train_denoised = pca.inverse_transform(X_train_denoised) # Geri oluştur (Reconstruction)
    
    X_test_denoised = pca.transform(X_test_h)
    X_test_denoised = pca.inverse_transform(X_test_denoised)
    
    # Tekrar Ölçekle ve Eğit
    scaler_denoise = StandardScaler()
    X_train_den_scaled = scaler_denoise.fit_transform(X_train_denoised)
    X_test_den_scaled = scaler_denoise.transform(X_test_denoised)
    
    mlp.fit(X_train_den_scaled, y_train)
    acc_j = mlp.score(X_test_den_scaled, y_test)
    print(f"j (Denoised) 50             50              {acc_j:.4f}          {1-acc_j:.4f}")

    # --- Görselleştirme (Orijinal vs Noisy) ---
    # Rastgele bir örnek seç
    sample_idx = random.randint(0, len(X)-1)
    sample_orig = X[sample_idx]
    
    # Sadece bu örnek için gürültü oluştur (Formül gereği)
    sample_noisy = sample_orig.copy()
    dims = np.random.choice(64, 10, replace=False)
    for d in dims:
        sample_noisy[d] = abs(sample_noisy[d] - 16)
        
    # Çizim
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Resimler 8x8 formatında olmalı
    axes[0].imshow(sample_orig.reshape(8, 8), cmap='gray_r')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(sample_noisy.reshape(8, 8), cmap='gray_r')
    axes[1].set_title("Noisy (|x-16|)")
    axes[1].axis('off')
    
    plt.savefig("Noising.pdf")
    print("\nGörsel 'Noising.pdf' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    main()