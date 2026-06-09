# Tubes KA - Problem 12 Manufaktur

Tugas besar mata kuliah kecerdasan artifisial.

Topik: Optimasi Penjadwalan Pemeliharaan Mesin

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
├── tahap 4
│   └── notebooks
├── setup_jupyter.py
└── requirements.txt
```

Dataset utama :

```text
D:\tubes_ka\ds12_manufaktur_maintenance.csv
```

## Instalasi Awal

Gunakan Python 3.10 atau versi yang lebih baru. Setelah repository selesai di-clone, masuk ke folder project:

```powershell
cd D:\tubes_ka
```

Opsional, buat virtual environment agar library project tidak bercampur dengan instalasi Python utama:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install library yang dibutuhkan:

```powershell
pip install -r requirements.txt
```

Pasang konfigurasi Jupyter agar folder `.ipynb_checkpoints` tidak dibuat:

```powershell
python setup_jupyter.py
```

Jika Jupyter sudah berjalan sebelum perintah tersebut dijalankan, tutup dulu server Jupyter lalu jalankan ulang.

## Menjalankan Jupyter

Jalankan Jupyter dari folder root project:

```powershell
cd D:\tubes_ka
jupyter notebook
```

Setelah halaman Jupyter terbuka, pilih notebook pada folder `tahap 1`, `tahap 2`, `tahap 3`, atau `tahap 4`.

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

## Menjalankan Tahap 4

```powershell
cd D:\tubes_ka
jupyter notebook "tahap 4\notebooks\Tubes_KA_Tahap4_Laporan_Akhir_Manufaktur.ipynb"
```

## File Kode Pendukung

- Tahap 1 notebook: `tahap 1\notebooks\Tubes_KA_Tahap1_Manufaktur.ipynb`
- Tahap 1 plot lampiran: `tahap 1\outputs\plots\korelasi_jam_operasi_vs_sisa_umur.png`
- Tahap 2 notebook: `tahap 2\notebooks\Tubes_KA_Tahap2_Desain_GA_Manufaktur.ipynb`
- Tahap 2 diagram: `tahap 2\outputs\plots\tahap2_arsitektur_integrasi_ga.png`
- Tahap 3 notebook: `tahap 3\notebooks\Tubes_KA_Tahap3_Integrasi_ML_GA_Manufaktur.ipynb`
- Tahap 3 plot konvergensi: `tahap 3\outputs\plots\tahap3_ga_convergence.png`
- Tahap 4 notebook: `tahap 4\notebooks\Tubes_KA_Tahap4_Laporan_Akhir_Manufaktur.ipynb`
