# task_visualization.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def main():
    # 1. Veri Seti Oluşturma
    # 500 örnek, 3 özellik (n_features=3), 2'si bilgilendirici (n_informative=2)
    X, y = make_classification(n_samples=500, n_features=3, n_redundant=0, 
                               n_informative=2, n_clusters_per_class=1, random_state=42)

    # 2. Eğitim ve Test Ayrımı (%70 Train, %30 Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. Modeli Eğitme
    clf = Perceptron(max_iter=100, tol=1e-3, random_state=42)
    clf.fit(X_train, y_train)

    # Katsayıları (Weights) ve Bias'ı al
    # Denklem: w1*x1 + w2*x2 + w3*x3 + b = 0
    w = clf.coef_[0]
    b = clf.intercept_[0]
    
    print(f"Model Katsayıları: w={w}, b={b}")

    # 4. 3D Çizim Hazırlığı
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Test verilerini çiz (Sınıf 0 için Mavi, Sınıf 1 için Kırmızı)
    # y_test==0 olanları seçip x1, x2, x3 koordinatlarını veriyoruz
    ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], X_test[y_test==0, 2], 
               c='blue', marker='^', label='Class 0')
    ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], X_test[y_test==1, 2], 
               c='red', marker='o', label='Class 1')

    # 5. Karar Düzlemini (Hyperplane) Oluşturma
    # x1 ve x2 için bir ızgara (grid) oluşturuyoruz
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.2),
                           np.arange(x2_min, x2_max, 0.2))

    # Düzlem denklemi: w1*x1 + w2*x2 + w3*x3 + b = 0
    # Buradan x3'ü çekersek: x3 = -(w1*x1 + w2*x2 + b) / w3
    if w[2] != 0:
        xx3 = -(w[0] * xx1 + w[1] * xx2 + b) / w[2]
        
        # Düzlemi çiz
        ax.plot_surface(xx1, xx2, xx3, alpha=0.3, color='yellow')
        print("Karar düzlemi çizildi.")
    else:
        print("w3 katsayısı 0 olduğu için 3D düzlem çizilemedi (dikey düzlem).")

    # Eksen etiketleri ve Başlık
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title('Perceptron 3D Decision Boundary')
    ax.legend()

    # Görseli göster
    plt.show()

if __name__ == "__main__":
    main()