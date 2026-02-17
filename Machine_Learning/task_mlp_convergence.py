# task_mlp_convergence.py

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def main():
    # 1. Digits veri setini yükle
    digits = load_digits()
    X, y = digits.data, digits.target

    # MLP için ölçeklendirme (Scaling) önemlidir, veriyi standartlaştırıyoruz
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Veriyi %70 Eğitim, %30 Test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # 3. MLP Ağını Kur (1 Gizli Katman, 50 Nöron, 100 İterasyon)
    # solver='sgd': Stokastik Gradyan İnişi (Hata düşüşünü net görmek için)
    # learning_rate_init: Öğrenme hızı (Grafiğin şeklini etkiler)
    mlp = MLPClassifier(hidden_layer_sizes=(50,), 
                        max_iter=100, 
                        solver='sgd', 
                        learning_rate_init=0.01,
                        random_state=42)

    # Modeli eğit
    print("Model eğitiliyor...")
    mlp.fit(X_train, y_train)

    # 4. Hata Değerlerini (Loss Curve) Çizdir
    loss_values = mlp.loss_curve_
    
    plt.figure(figsize=(8, 6))
    plt.plot(loss_values, color='red', linewidth=2)
    plt.title('Convergence of error with MLP')
    plt.xlabel('Iteration')
    plt.ylabel('Error (Loss)')
    plt.grid(True)
    
    # Grafiği kaydet ve göster
    plt.savefig("Error_Convergence.pdf")
    print("Grafik 'Error_Convergence.pdf' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    main()