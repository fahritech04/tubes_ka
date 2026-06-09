from pathlib import Path
import json
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

ROOT=Path(r'D:\tubes_ka')
STAGE=ROOT/'tahap 1'
OUT=STAGE/'outputs'
METRICS=OUT/'metrics'
DATAOUT=OUT/'data'
MODELS=STAGE/'models'
for folder in [METRICS, DATAOUT, MODELS]:
    folder.mkdir(parents=True, exist_ok=True)

csv=ROOT/'ds12_manufaktur_maintenance.csv'
df=pd.read_csv(csv)

# Fitur untuk ML RUL: sensor/operasional mesin. id_log_sensor adalah identifier; durasi perbaikan dipakai GA, bukan prediktor RUL.
target='sisa_umur_jam'
features=['jam_operasi_aktif','jenis_mesin','tekanan_oli_bar','suhu_c','vibrasi_mm_s','shift_operator','kebisingan_lingkungan_db']
X=df[features]
y=df[target]

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1764705882, random_state=42) # 0.15 / 0.85

num_cols=X.select_dtypes(include=['number']).columns.tolist()
cat_cols=X.select_dtypes(exclude=['number']).columns.tolist()

numeric_scaled=Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
numeric_tree=Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
cat=Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

pre_scaled=ColumnTransformer([('num', numeric_scaled, num_cols),('cat', cat, cat_cols)])
pre_tree=ColumnTransformer([('num', numeric_tree, num_cols),('cat', cat, cat_cols)])

models={
    'Dummy Mean': Pipeline([('preprocess', pre_tree),('model', DummyRegressor(strategy='mean'))]),
    'Ridge Regression': Pipeline([('preprocess', pre_scaled),('model', Ridge(alpha=1.0, random_state=42))]),
    'Decision Tree': Pipeline([('preprocess', pre_tree),('model', DecisionTreeRegressor(max_depth=8, random_state=42))]),
    'Random Forest': Pipeline([('preprocess', pre_tree),('model', RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=2, random_state=42, n_jobs=-1))]),
    'Gradient Boosting': Pipeline([('preprocess', pre_tree),('model', GradientBoostingRegressor(n_estimators=180, learning_rate=0.06, max_depth=3, random_state=42))]),
}

def metrics(model, Xs, ys):
    pred=model.predict(Xs)
    return {
        'RMSE': float(root_mean_squared_error(ys, pred)),
        'MAE': float(mean_absolute_error(ys, pred)),
        'R2': float(r2_score(ys, pred)),
    }

rows=[]
best_name=None
best_rmse=1e18
for name, model in models.items():
    model.fit(X_train, y_train)
    mval=metrics(model, X_val, y_val)
    mtest=metrics(model, X_test, y_test)
    rows.append({'model': name, **{f'val_{k}': v for k,v in mval.items()}, **{f'test_{k}': v for k,v in mtest.items()}})
    if mval['RMSE'] < best_rmse:
        best_rmse=mval['RMSE']; best_name=name

res=pd.DataFrame(rows).sort_values('val_RMSE')
res.to_csv(METRICS/'tahap1_model_metrics.csv', index=False)

best=models[best_name]
joblib.dump(best, MODELS/'best_rul_model.joblib')

# EDA summary
summary={
    'shape': df.shape,
    'missing': df.isna().sum().to_dict(),
    'numeric_corr_pearson': df.select_dtypes(include='number').corr(numeric_only=True)[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False).to_dict(),
    'jenis_mesin_summary': df.groupby('jenis_mesin')[target].agg(['count','mean','std','min','max']).round(3).to_dict('index'),
    'shift_summary': df.groupby('shift_operator')[target].agg(['count','mean','std']).round(3).to_dict('index'),
    'best_model': best_name,
}
(METRICS/'tahap1_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

# Input antrean perbaikan: 50 mesin anomali dengan risiko kerusakan tertinggi.
urgent=df.nsmallest(50, target).copy()
urgent_X=urgent[features]
urgent['prediksi_sisa_umur_jam']=best.predict(urgent_X).round(2)
urgent[['id_log_sensor','jenis_mesin','sisa_umur_jam','prediksi_sisa_umur_jam','durasi_perbaikan_jam','jam_operasi_aktif','suhu_c','vibrasi_mm_s','tekanan_oli_bar']].to_csv(DATAOUT/'input_50_mesin_anomali_prediksi.csv', index=False)

print(res.round(4).to_string(index=False))
print('\nModel terbaik:', best_name)
print('Output Tahap 1:', METRICS/'tahap1_model_metrics.csv', METRICS/'tahap1_summary.json', MODELS/'best_rul_model.joblib')
