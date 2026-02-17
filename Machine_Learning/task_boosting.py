# task_boosting.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from matplotlib.colors import ListedColormap

def main():
    # 1. Moon Veri Seti Oluşturma
    # n_samples > 100 (örneğin 500), noise=0.2
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

    # 2. Eğitim/Test Ayrımı (%70 - %30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 3. Temel Sınıflandırıcı (Logistic Regression with SGD)
    # loss='log_loss' (veya eski sürümlerde 'log') Logistic Regression demektir.
    base_estimator = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)

    # 4. AdaBoost Classifier
    # n_estimators=4: 4 tane ardışık model eğitilecek
    # algorithm='SAMME': Ayrık (discrete) boosting için genelde kullanılır, 
    # ancak varsayılan SAMME.R de olur. Görselleştirme için basit tutuyoruz.
    adaboost = AdaBoostClassifier(estimator=base_estimator, 
                                  n_estimators=4, 
                                  algorithm='SAMME',
                                  random_state=42)
    
    adaboost.fit(X_train, y_train)

    # 5. Görselleştirme (1x4 Subplot)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Meshgrid oluştur (Arka plan renklendirmesi için)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Renk haritaları
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Eğitilmiş her bir "zayıf" öğrenciyi (estimator) sırayla çiz
    for i, estimator in enumerate(adaboost.estimators_):
        ax = axes[i]
        
        # Karar sınırını hesapla
        if hasattr(estimator, "decision_function"):
            Z = estimator.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = estimator.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        Z = Z.reshape(xx.shape)

        # Kontur çizimi (Decision Boundary)
        # Kesikli çizgi (dashed) ile sınırı gösterelim
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', linestyles='dashed')
        
        # Test verilerini çiz
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)
        
        ax.set_title(f"Learner #{i+1}")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

    # Kaydet ve Göster
    plt.tight_layout()
    plt.savefig("BaseLearner_Visualization.pdf")
    print("Görsel 'BaseLearner_Visualization.pdf' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    main()