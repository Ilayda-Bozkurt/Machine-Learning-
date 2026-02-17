# task_ensemble_hyperps.py

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def main():
    # 1. Veri Yükleme
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 2. Eğitim/Test Ayrımı (%70 - %30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    estimators = []
    
    # 3. 10 Farklı MLP Örneği Oluşturma
    # h: 1'den 10'a kadar
    for h in range(1, 11):
        # Katman yapısı: 2^h, 2^(h-1), ..., 2^1
        hidden_layers = tuple([2**i for i in range(h, 0, -1)])
        
        # Modeli Pipeline içine koyuyoruz (Scaling + MLP)
        # MLP'nin yakınsaması için StandardScaler şarttır.
        mlp = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=hidden_layers, 
                          max_iter=1000, # Yakınsama için yeterli süre
                          random_state=42)
        )
        
        # Estimator listesine ekle (İsim, Model)
        model_name = f"1#{h}"
        estimators.append((model_name, mlp))

    # 4. Bireysel Performansları Ölçme ve Yazdırma
    print("-" * 40)
    for name, model in estimators:
        # Modeli eğit
        model.fit(X_train, y_train)
        # Test et
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # İstenen çıktı formatı
        print(f"Parameter setting: {name} Accuracy: {acc}")

    # 5. Ensemble Learning (Voting) Uygulama
    # voting='hard': Çoğunluk oyu (Sınıf tahmini üzerinden)
    # Not: VotingClassifier eğitilmiş modelleri clone'layıp baştan eğitebilir, 
    # biz zaten yukarıda eğittik ama VotingClassifier'ın fit metodunu çağırmak standarttır.
    ensemble_clf = VotingClassifier(estimators=estimators, voting='hard')
    ensemble_clf.fit(X_train, y_train)
    
    ensemble_acc = ensemble_clf.score(X_test, y_test)

    print("-" * 40)
    print(f"Ensemble Learning Accuracy: {ensemble_acc}")

if __name__ == "__main__":
    main()