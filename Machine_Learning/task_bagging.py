# task_bagging.py

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def main():
    # 1. Veri Yükleme ve Hazırlama
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Ölçeklendirme (MLP için kritik)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Eğitim/Test Ayrımı (%70 - %30)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    test_sample_count = len(y_test) # Toplam test örneği sayısı (örn: 540)

    # 3. Temel Sınıflandırıcıyı (Base Learner) Tanımla
    # İstenen yapı: 4 gizli katman (16, 8, 4, 2), max_iter >= 1000
    base_mlp = MLPClassifier(hidden_layer_sizes=(16, 8, 4, 2), 
                             max_iter=1000, 
                             random_state=42)

    # 4. Bagging Classifier Tanımla
    # n_estimators=8: 8 tane base classifier olacak.
    # max_samples=0.125: Her biri verinin 1/8'ini kullanacak (1/8 = 0.125).
    # bootstrap=True: Örnekleme yerine koymalı (Replacement).
    bagging_clf = BaggingClassifier(estimator=base_mlp, 
                                    n_estimators=8, 
                                    max_samples=0.125, 
                                    bootstrap=True, 
                                    random_state=42)

    print("Bagging modeli eğitiliyor (biraz zaman alabilir)...")
    bagging_clf.fit(X_train, y_train)

    # 5. Her bir alt öğrencinin (learner) performansını ölç
    # bagging_clf.estimators_ listesi eğitilmiş 8 modeli tutar.
    print("-" * 60)
    for i, learner in enumerate(bagging_clf.estimators_):
        # Her learner için test setini tahmin et
        # Not: Sklearn'de base learnerlar ölçeklenmiş veri bekler
        # Ancak Bagging içindeki learnerlar bazen subset feature kullanır. 
        # Burada feature subsetting (max_features) varsayılan 1.0 olduğu için tüm featureları verebiliriz.
        
        # Bagging içindeki modelleri manuel tahmin ettiriyoruz
        y_pred_learner = learner.predict(X_test)
        
        # Doğru sayısını hesapla (normalize=False tam sayıyı verir)
        correct_count = accuracy_score(y_test, y_pred_learner, normalize=False)
        
        print(f"{correct_count} out of {test_sample_count} instances are correctly classified by learner #{i+1}")

    # 6. Bagging (Topluluk) Performansını Ölç
    y_pred_bagging = bagging_clf.predict(X_test)
    bagging_correct_count = accuracy_score(y_test, y_pred_bagging, normalize=False)
    
    print("-" * 60)
    print(f"{bagging_correct_count} out of {test_sample_count} instances are correctly classified by bagging")

if __name__ == "__main__":
    main()