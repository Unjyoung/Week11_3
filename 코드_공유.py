# ===== 라이브러리 및 데이터 불러오기 =====

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

# 와인 데이터 로드 및 DataFrame 생성
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# 'target' 컬럼 drop하여 feature(X)와 label(y) 분리
X = df.drop('target', axis=1)
y = df['target']

# 데이터 분할 (test size = 0.2, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


####### A 작업자 작업 수행 #######

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Decision Tree 모델 생성 및 학습
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 예측 및 평가
y_pred_dt = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print(f"[A 작업자] Decision Tree Accuracy: {dt_accuracy:.4f}")
