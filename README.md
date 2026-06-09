# Tubes KA - Problem 12 Manufaktur

Tugas besar mata kuliah kecerdasan artifisial

## Struktur Folder

```text
D:\tubes_ka
├── tahap 1
│   ├── notebooks
│   ├── models
│   ├── outputs
│   └── scripts
├── tahap 2
│   ├── notebooks
│   ├── outputs
│   └── scripts
├── tahap 3
│   ├── notebooks
│   ├── outputs
│   └── scripts
└── requirements.txt
```

Dataset utama :

```text
D:\tubes_ka\ds12_manufaktur_maintenance.csv
```

## Menjalankan Tahap 1

```powershell
cd D:\tubes_ka
python "tahap 1\scripts\tahap1_train_baseline.py"
python "tahap 1\scripts\tahap1_generate_plots.py"
jupyter notebook "tahap 1\notebooks\Tubes_KA_Tahap1_Manufaktur.ipynb"
```

## Menjalankan Tahap 2

```powershell
cd D:\tubes_ka
python "tahap 2\scripts\tahap2_desain_ga.py"
jupyter notebook "tahap 2\notebooks\Tubes_KA_Tahap2_Desain_GA_Manufaktur.ipynb"
```

## Menjalankan Tahap 3

```powershell
cd D:\tubes_ka
python "tahap 3\scripts\tahap3_integration_ga.py"
jupyter notebook "tahap 3\notebooks\Tubes_KA_Tahap3_Integrasi_ML_GA_Manufaktur.ipynb"
```

## File Kode Pendukung

- Tahap 1 notebook: `tahap 1\notebooks\Tubes_KA_Tahap1_Manufaktur.ipynb`
- Tahap 1 plot lampiran: `tahap 1\outputs\plots\korelasi_jam_operasi_vs_sisa_umur.png`
- Tahap 2 notebook: `tahap 2\notebooks\Tubes_KA_Tahap2_Desain_GA_Manufaktur.ipynb`
- Tahap 2 diagram: `tahap 2\outputs\plots\tahap2_arsitektur_integrasi_ga.png`
- Tahap 3 notebook: `tahap 3\notebooks\Tubes_KA_Tahap3_Integrasi_ML_GA_Manufaktur.ipynb`
- Tahap 3 plot konvergensi: `tahap 3\outputs\plots\tahap3_ga_convergence.png`
