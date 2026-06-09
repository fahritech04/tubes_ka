from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT = Path(r'D:\tubes_ka')
STAGE1 = ROOT / 'tahap 1'
STAGE2 = ROOT / 'tahap 2'
OUT = STAGE2 / 'outputs'
DATAOUT = OUT / 'data'
DATAOUT.mkdir(parents=True, exist_ok=True)

src = STAGE1 / 'outputs' / 'data' / 'input_50_mesin_anomali_prediksi.csv'
df = pd.read_csv(src)

# GA tidak boleh memakai RUL negatif; prediksi negatif berarti mesin sudah sangat kritis.
df['rul_ga_jam'] = df['prediksi_sisa_umur_jam'].clip(lower=0).round(2)
df['machine_code'] = [f'M{i:02d}' for i in range(1, len(df) + 1)]

def evaluate_schedule(order, data):
    """Evaluate one permutation chromosome.

    order berisi index baris data. Mesin dianggap breakdown jika waktu mulai
    perbaikan lebih besar dari RUL prediksi mesin tersebut.
    """
    seen = set(order)
    if len(order) != len(data) or len(seen) != len(data) or seen != set(range(len(data))):
        raise ValueError('Kromosom tidak valid: harus berupa permutasi lengkap tanpa duplikasi.')

    time_now = 0.0
    breakdown_count = 0
    total_lateness = 0.0
    total_waiting = 0.0
    rows = []

    for seq, idx in enumerate(order, start=1):
        row = data.iloc[idx]
        start = time_now
        finish = start + float(row['durasi_perbaikan_jam'])
        rul = float(row['rul_ga_jam'])
        lateness = max(0.0, start - rul)
        breakdown = int(start > rul)
        breakdown_count += breakdown
        total_lateness += lateness
        total_waiting += start
        rows.append({
            'urutan': seq,
            'machine_code': row['machine_code'],
            'id_log_sensor': row['id_log_sensor'],
            'jenis_mesin': row['jenis_mesin'],
            'rul_ga_jam': rul,
            'durasi_perbaikan_jam': int(row['durasi_perbaikan_jam']),
            'mulai_jam': start,
            'selesai_jam': finish,
            'breakdown_saat_menunggu': breakdown,
            'keterlambatan_jam': lateness,
        })
        time_now = finish

    # Cost diminimasi; breakdown dibuat dominan, lateness sebagai tie-breaker.
    cost = 10000 * breakdown_count + 100 * total_lateness + 0.01 * total_waiting
    fitness = 1 / (1 + cost)
    return {
        'breakdown_count': int(breakdown_count),
        'total_lateness_jam': round(total_lateness, 2),
        'total_waiting_jam': round(total_waiting, 2),
        'makespan_jam': round(time_now, 2),
        'cost': round(cost, 4),
        'fitness': fitness,
        'schedule': pd.DataFrame(rows),
    }

orders = {
    'FCFS_input_awal': list(range(len(df))),
    'SPT_durasi_terpendek': df.sort_values('durasi_perbaikan_jam').index.tolist(),
    'EDD_RUL_terpendek': df.sort_values(['rul_ga_jam', 'durasi_perbaikan_jam']).index.tolist(),
}

summary_rows = []
best_name = None
best_cost = float('inf')
for name, order in orders.items():
    result = evaluate_schedule(order, df)
    summary_rows.append({
        'jadwal': name,
        'breakdown_count': result['breakdown_count'],
        'total_lateness_jam': result['total_lateness_jam'],
        'total_waiting_jam': result['total_waiting_jam'],
        'makespan_jam': result['makespan_jam'],
        'cost': result['cost'],
        'fitness': result['fitness'],
    })
    if result['cost'] < best_cost:
        best_cost = result['cost']
        best_name = name
        result['schedule'].to_csv(DATAOUT / 'tahap2_jadwal_pembanding_terbaik.csv', index=False)

summary = pd.DataFrame(summary_rows).sort_values('cost')
summary.to_csv(DATAOUT / 'tahap2_evaluasi_fitness_jadwal.csv', index=False)

ga_input_cols = [
    'machine_code', 'id_log_sensor', 'jenis_mesin', 'prediksi_sisa_umur_jam',
    'rul_ga_jam', 'durasi_perbaikan_jam', 'jam_operasi_aktif', 'suhu_c',
    'vibrasi_mm_s', 'tekanan_oli_bar'
]
df[ga_input_cols].to_csv(DATAOUT / 'tahap2_input_ga_50_mesin.csv', index=False)

kromosom_awal = df.sort_values(['rul_ga_jam', 'durasi_perbaikan_jam'])['machine_code'].head(10).tolist()
desain_ga = {
    'arsitektur': 'Opsi A - Pra-Komputasi',
    'output_ml_untuk_ga': 'rul_ga_jam = max(0, prediksi_sisa_umur_jam)',
    'tipe_kromosom': 'Permutasi 50 machine_code',
    'kromosom_awal_10_gen_pertama': kromosom_awal,
    'jadwal_pembanding_terbaik': best_name,
    'rumus_fitness': 'fitness = 1 / (1 + 10000*breakdown_count + 100*total_lateness_jam + 0.01*total_waiting_jam)',
    'batasan_mutlak': [
        'Setiap machine_code muncul tepat satu kali.',
        'Tidak boleh ada duplikasi atau mesin hilang.',
        'Satu tim teknisi mengerjakan satu mesin pada satu waktu.',
        'Perbaikan bersifat non-preemptive: satu perbaikan diselesaikan sebelum pindah ke mesin berikutnya.'
    ],
    'crossover': 'Order Crossover (OX) atau Partially Matched Crossover (PMX)',
    'mutasi': 'Swap mutation dan/atau inversion mutation',
}
(DATAOUT / 'tahap2_desain_ga.json').write_text(json.dumps(desain_ga, indent=2), encoding='utf-8')

print(summary.round(6).to_string(index=False))
print('\nKromosom awal 10 gen pertama:', kromosom_awal)
print('Output Tahap 2 tersimpan di', DATAOUT)
