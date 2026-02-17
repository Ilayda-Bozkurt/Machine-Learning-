# task_mlp_hidden_layer.py

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def main():
    # 1. Veri Yükleme
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Ölçeklendirme (Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Eğitim/Test Ayrımı (%70 - %30)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    train_scores = []
    test_scores = []
    # 1'den 10'a kadar katman sayılarını deneyeceğiz (Figure 6'daki eksen 10'a kadar)
    layer_sizes_range = range(1, 11)

    print(f"{'h (Katman)':<10} {'Yapı (Nöronlar)':<35} {'Train Acc':<10} {'Test Acc':<10}")
    print("-" * 80)

    for h in layer_sizes_range:
        # 3. Dinamik Nöron Yapısı Oluşturma
        # Kural: 2^h, 2^(h-1), ..., 2^1
        # Örn h=3 için: (2^3, 2^2, 2^1) -> (8, 4, 2)
        hidden_layers = tuple([2**i for i in range(h, 0, -1)])
        
        # Modeli oluştur
        # Katman sayısı arttıkça eğitmek zorlaşır, max_iter'i yüksek tutuyoruz (2000)
        mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, 
                            max_iter=2000, 
                            random_state=42)
        
        # Eğit
        mlp.fit(X_train, y_train)
        
        # Skorları hesapla
        tr_score = mlp.score(X_train, y_train)
        te_score = mlp.score(X_test, y_test)
        
        train_scores.append(tr_score)
        test_scores.append(te_score)
        
        print(f"{h:<10} {str(hidden_layers):<35} {tr_score:.4f}     {te_score:.4f}")

    # 4. Grafik Çizimi (Mavi: Train, Kırmızı: Test)
    plt.figure(figsize=(10, 6))
    plt.plot(layer_sizes_range, train_scores, marker='*', color='blue', label='Train', linewidth=2)
    plt.plot(layer_sizes_range, test_scores, marker='+', color='red', label='Test', linewidth=2)
    
    plt.title('Train & test scores as a function of hidden layer size')
    plt.xlabel('Hidden Layer Size (h)')
    plt.ylabel('Accuracy Score')
    plt.xticks(layer_sizes_range)
    plt.legend()
    plt.grid(True)
    
    # Kaydet ve Göster
    plt.savefig("Hidden_Layer_Analysis.pdf")
    print("\nGrafik 'Hidden_Layer_Analysis.pdf' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    main()