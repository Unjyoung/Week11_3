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
















####### B 작업자 작업 수행 #######

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# XGBClassifier 모델 생성 및 학습
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(X_train, y_train)

# 예측 및 평가
y_pred_xgb = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

print(f"[B 작업자] XGBoost Accuracy: {xgb_accuracy:.4f}")
