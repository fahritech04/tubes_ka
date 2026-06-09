from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

ROOT=Path(r'D:\tubes_ka')
STAGE=ROOT/'tahap 1'
OUT=STAGE/'outputs'
METRICS=OUT/'metrics'
PLOTS=OUT/'plots'
for folder in [METRICS, PLOTS]:
    folder.mkdir(parents=True, exist_ok=True)

csv=ROOT/'ds12_manufaktur_maintenance.csv'
df=pd.read_csv(csv)
target='sisa_umur_jam'
features=['jam_operasi_aktif','jenis_mesin','tekanan_oli_bar','suhu_c','vibrasi_mm_s','shift_operator','kebisingan_lingkungan_db']

sns.set_theme(style='whitegrid', palette='deep')

# 1. Scatter korelasi utama: jam operasi vs RUL
plt.figure(figsize=(8,5))
sns.regplot(data=df.sample(4000, random_state=42), x='jam_operasi_aktif', y=target, scatter_kws={'alpha':0.18, 's':14}, line_kws={'color':'#c1121f', 'linewidth':2})
plt.title('Korelasi Jam Operasi Aktif terhadap Sisa Umur Mesin')
plt.xlabel('Jam Operasi Aktif')
plt.ylabel('Sisa Umur (jam)')
plt.tight_layout()
plt.savefig(PLOTS/'korelasi_jam_operasi_vs_sisa_umur.png', dpi=180)
plt.close()

# 2. Scatter vibrasi vs RUL
plt.figure(figsize=(8,5))
sns.regplot(data=df.sample(4000, random_state=7), x='vibrasi_mm_s', y=target, scatter_kws={'alpha':0.18, 's':14}, line_kws={'color':'#c1121f', 'linewidth':2})
plt.title('Korelasi Vibrasi terhadap Sisa Umur Mesin')
plt.xlabel('Vibrasi (mm/s)')
plt.ylabel('Sisa Umur (jam)')
plt.tight_layout()
plt.savefig(PLOTS/'korelasi_vibrasi_vs_sisa_umur.png', dpi=180)
plt.close()

# 3. Heatmap korelasi numerik
corr=df.select_dtypes(include='number').corr(numeric_only=True)
plt.figure(figsize=(9,7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, cbar_kws={'shrink':0.8})
plt.title('Heatmap Korelasi Fitur Numerik')
plt.tight_layout()
plt.savefig(PLOTS/'heatmap_korelasi_numerik.png', dpi=180)
plt.close()

# 4. Predicted vs actual for best model
best=joblib.load(STAGE/'models'/'best_rul_model.joblib')
X=df[features]
y=df[target]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
pred=best.predict(X_test)
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=pred, alpha=0.35, s=18)
lo=min(y_test.min(), pred.min()); hi=max(y_test.max(), pred.max())
plt.plot([lo,hi],[lo,hi], color='#c1121f', linewidth=2)
plt.title('Prediksi vs Aktual RUL - Gradient Boosting')
plt.xlabel('Aktual sisa umur (jam)')
plt.ylabel('Prediksi sisa umur (jam)')
plt.tight_layout()
plt.savefig(PLOTS/'prediksi_vs_aktual_gradient_boosting.png', dpi=180)
plt.close()

# 5. Permutation importance on test subset
perm=permutation_importance(best, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1, scoring='neg_root_mean_squared_error')
imp=pd.DataFrame({'feature':features, 'importance_rmse_increase':perm.importances_mean, 'std':perm.importances_std}).sort_values('importance_rmse_increase', ascending=False)
imp.to_csv(METRICS/'tahap1_permutation_importance.csv', index=False)
plt.figure(figsize=(8,5))
sns.barplot(data=imp, y='feature', x='importance_rmse_increase', color='#2a9d8f')
plt.title('Permutation Importance Model Terbaik')
plt.xlabel('Kenaikan RMSE saat fitur diacak (semakin besar semakin penting)')
plt.ylabel('Fitur')
plt.tight_layout()
plt.savefig(PLOTS/'permutation_importance_gradient_boosting.png', dpi=180)
plt.close()

print('Plot Tahap 1 tersimpan di', PLOTS)
print(imp.round(4).to_string(index=False))
