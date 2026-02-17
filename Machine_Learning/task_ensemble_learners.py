# task_ensemble_learners.py

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def main():
    # 1. Veri Yükleme
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 2. Üç Farklı Sınıflandırıcı Oluşturma
    # Not: SVM ve KNN uzaklık temelli olduğu için ölçeklendirme (Scaling) gerektirir.
    # Bu yüzden pipeline kullanıyoruz.
    
    # Learner 1: Logistic Regression (Basit ve etkili)
    clf1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    
    # Learner 2: Decision Tree (Ağaç tabanlı, ölçekleme gerektirmez ama pipeline zarar vermez)
    clf2 = DecisionTreeClassifier(random_state=42)
    
    # Learner 3: SVM (Güçlü bir algoritma)
    clf3 = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))

    estimators = [
        ('lr', clf1),
        ('dt', clf2),
        ('svm', clf3)
    ]

    # 3. Ensemble (Voting) Oluşturma
    # voting='soft': Olasılık ortalamasına bakar (Genelde hard voting'den daha iyidir)
    ensemble_clf = VotingClassifier(estimators=estimators, voting='soft')

    # 4. 5-Fold Cross Validation ile Performans Ölçümü
    # StratifiedKFold sınıf dengesizliğini korur
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("5-Fold Cross Validation sonuçları hesaplanıyor...\n")

    # Her bir learner için skorları hesapla
    for i, (name, clf) in enumerate(estimators):
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        mean_acc = scores.mean()
        print(f"Accuracy obtained by learner #{i+1} ({name}) is: {mean_acc:.4f}")

    # Ensemble için skoru hesapla
    ensemble_scores = cross_val_score(ensemble_clf, X, y, cv=cv, scoring='accuracy')
    ensemble_acc = ensemble_scores.mean()
    
    print("-" * 50)
    print(f"Accuracy obtained by ensemble learner is: {ensemble_acc:.4f}")

if __name__ == "__main__":
    main()