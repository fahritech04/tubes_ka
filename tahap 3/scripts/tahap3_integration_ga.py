from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

ROOT = Path(r'D:\tubes_ka')
STAGE1 = ROOT / 'tahap 1'
STAGE3 = ROOT / 'tahap 3'
OUT = STAGE3 / 'outputs'
DATAOUT = OUT / 'data'
PLOTS = OUT / 'plots'
for folder in [DATAOUT, PLOTS]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MODEL_PATH = STAGE1 / 'models' / 'best_rul_model.joblib'
SOURCE_CSV = ROOT / 'ds12_manufaktur_maintenance.csv'
SIM50_PATH = STAGE1 / 'outputs' / 'data' / 'input_50_mesin_anomali_prediksi.csv'

FEATURES = [
    'jam_operasi_aktif', 'jenis_mesin', 'tekanan_oli_bar', 'suhu_c',
    'vibrasi_mm_s', 'shift_operator', 'kebisingan_lingkungan_db'
]

# Interface ML -> GA: ambil 50 mesin, prediksi RUL dari model ML, lalu bentuk tabel GA.
model = joblib.load(MODEL_PATH)
source_df = pd.read_csv(SOURCE_CSV)
sim50_ids = pd.read_csv(SIM50_PATH)['id_log_sensor'].tolist()
ml_input = source_df[source_df['id_log_sensor'].isin(sim50_ids)].copy()
ml_input['order_ref'] = ml_input['id_log_sensor'].map({mid: i for i, mid in enumerate(sim50_ids)})
ml_input = ml_input.sort_values('order_ref').drop(columns=['order_ref']).reset_index(drop=True)

predicted_rul = model.predict(ml_input[FEATURES])
ga_df = ml_input.copy()
ga_df['machine_code'] = [f'M{i:02d}' for i in range(1, len(ga_df) + 1)]
ga_df['prediksi_sisa_umur_jam'] = np.round(predicted_rul, 2)
ga_df['rul_ga_jam'] = np.clip(ga_df['prediksi_sisa_umur_jam'], a_min=0, a_max=None).round(2)

interface_cols = [
    'machine_code', 'id_log_sensor', 'jenis_mesin', 'jam_operasi_aktif', 'suhu_c',
    'vibrasi_mm_s', 'tekanan_oli_bar', 'shift_operator', 'kebisingan_lingkungan_db',
    'durasi_perbaikan_jam', 'prediksi_sisa_umur_jam', 'rul_ga_jam'
]
ga_df[interface_cols].to_csv(DATAOUT / 'tahap3_interface_ml_to_ga.csv', index=False)

N = len(ga_df)
DURATIONS = ga_df['durasi_perbaikan_jam'].astype(float).to_numpy()
RUL = ga_df['rul_ga_jam'].astype(float).to_numpy()


def validate_chromosome(chromosome):
    return len(chromosome) == N and len(set(chromosome)) == N and set(chromosome) == set(range(N))


def evaluate_chromosome(chromosome):
    if not validate_chromosome(chromosome):
        return {
            'breakdown_count': N,
            'total_lateness_jam': 10_000.0,
            'total_waiting_jam': 10_000.0,
            'makespan_jam': float(DURATIONS.sum()),
            'cost': 999_999_999.0,
            'fitness': 1 / (1 + 999_999_999.0),
        }

    time_now = 0.0
    breakdown_count = 0
    total_lateness = 0.0
    total_waiting = 0.0

    for idx in chromosome:
        start = time_now
        lateness = max(0.0, start - RUL[idx])
        if start > RUL[idx]:
            breakdown_count += 1
        total_lateness += lateness
        total_waiting += start
        time_now += DURATIONS[idx]

    cost = 10000 * breakdown_count + 100 * total_lateness + 0.01 * total_waiting
    fitness = 1 / (1 + cost)
    return {
        'breakdown_count': int(breakdown_count),
        'total_lateness_jam': float(total_lateness),
        'total_waiting_jam': float(total_waiting),
        'makespan_jam': float(time_now),
        'cost': float(cost),
        'fitness': float(fitness),
    }


def schedule_dataframe(chromosome):
    rows = []
    time_now = 0.0
    for seq, idx in enumerate(chromosome, start=1):
        row = ga_df.iloc[idx]
        start = time_now
        finish = start + float(row['durasi_perbaikan_jam'])
        lateness = max(0.0, start - float(row['rul_ga_jam']))
        rows.append({
            'urutan': seq,
            'machine_code': row['machine_code'],
            'id_log_sensor': row['id_log_sensor'],
            'jenis_mesin': row['jenis_mesin'],
            'rul_ga_jam': row['rul_ga_jam'],
            'durasi_perbaikan_jam': int(row['durasi_perbaikan_jam']),
            'mulai_jam': round(start, 2),
            'selesai_jam': round(finish, 2),
            'breakdown_saat_menunggu': int(start > float(row['rul_ga_jam'])),
            'keterlambatan_jam': round(lateness, 2),
        })
        time_now = finish
    return pd.DataFrame(rows)


def order_crossover(parent_a, parent_b):
    left, right = sorted(random.sample(range(N), 2))
    child = [-1] * N
    child[left:right + 1] = parent_a[left:right + 1]
    used = set(child[left:right + 1])
    fill_values = [gene for gene in parent_b if gene not in used]
    fill_positions = [i for i, gene in enumerate(child) if gene == -1]
    for pos, gene in zip(fill_positions, fill_values):
        child[pos] = gene
    return child


def mutate(chromosome, mutation_rate=0.2):
    child = chromosome[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(N), 2)
        child[i], child[j] = child[j], child[i]
    if random.random() < mutation_rate / 2:
        i, j = sorted(random.sample(range(N), 2))
        child[i:j + 1] = reversed(child[i:j + 1])
    return child


def tournament_select(population, scores, tournament_size=3):
    contestants = random.sample(range(len(population)), tournament_size)
    best_idx = max(contestants, key=lambda idx: scores[idx]['fitness'])
    return population[best_idx][:]


def initial_population(population_size):
    population = []
    # Seed a few reasonable schedules to stabilize the run.
    population.append(list(np.argsort(RUL)))
    population.append(list(np.argsort(DURATIONS)))
    population.append(list(np.lexsort((DURATIONS, RUL))))
    ratio_order = sorted(range(N), key=lambda i: (RUL[i] / max(DURATIONS[i], 1e-9), RUL[i]))
    population.append(ratio_order)

    while len(population) < population_size:
        candidate = list(range(N))
        random.shuffle(candidate)
        population.append(candidate)
    return population


def run_ga(population_size=100, generations=60, crossover_rate=0.85, mutation_rate=0.22, elitism=2):
    population = initial_population(population_size)
    history = []
    global_best = None
    global_best_score = None

    for generation in range(generations + 1):
        scores = [evaluate_chromosome(chrom) for chrom in population]
        best_idx = max(range(len(population)), key=lambda i: scores[i]['fitness'])
        best_score = scores[best_idx]
        if global_best_score is None or best_score['fitness'] > global_best_score['fitness']:
            global_best = population[best_idx][:]
            global_best_score = best_score.copy()

        history.append({
            'generation': generation,
            'best_fitness': global_best_score['fitness'],
            'best_cost': global_best_score['cost'],
            'best_breakdown_count': global_best_score['breakdown_count'],
            'best_total_lateness_jam': global_best_score['total_lateness_jam'],
            'generation_mean_fitness': float(np.mean([s['fitness'] for s in scores])),
            'generation_best_fitness': best_score['fitness'],
            'generation_best_cost': best_score['cost'],
        })

        if generation == generations:
            break

        elite_indices = sorted(range(len(population)), key=lambda i: scores[i]['fitness'], reverse=True)[:elitism]
        next_population = [population[i][:] for i in elite_indices]
        while len(next_population) < population_size:
            parent_a = tournament_select(population, scores)
            parent_b = tournament_select(population, scores)
            if random.random() < crossover_rate:
                child = order_crossover(parent_a, parent_b)
            else:
                child = parent_a[:]
            child = mutate(child, mutation_rate=mutation_rate)
            next_population.append(child)
        population = next_population

    return global_best, global_best_score, pd.DataFrame(history)


best_chromosome, best_score, history = run_ga()
history.to_csv(DATAOUT / 'tahap3_ga_convergence_history.csv', index=False)
best_schedule = schedule_dataframe(best_chromosome)
best_schedule.to_csv(DATAOUT / 'tahap3_best_schedule_ga.csv', index=False)

best_summary = {
    'random_seed': RANDOM_SEED,
    'population_size': 100,
    'generations': 60,
    'crossover': 'Order Crossover (OX)',
    'mutation': 'swap + inversion mutation',
    'selection': 'Tournament selection size 3 + elitism 2',
    'interface_input_shape': list(ml_input[FEATURES].shape),
    'interface_output_shape': [len(predicted_rul)],
    'ga_chromosome_length': N,
    'best_score': {
        'fitness': best_score['fitness'],
        'cost': round(best_score['cost'], 4),
        'breakdown_count': best_score['breakdown_count'],
        'total_lateness_jam': round(best_score['total_lateness_jam'], 2),
        'total_waiting_jam': round(best_score['total_waiting_jam'], 2),
        'makespan_jam': round(best_score['makespan_jam'], 2),
    },
    'best_chromosome_machine_code_first_15': best_schedule['machine_code'].head(15).tolist(),
}
(DATAOUT / 'tahap3_integration_summary.json').write_text(json.dumps(best_summary, indent=2), encoding='utf-8')

# Plot konvergensi: cost turun dan breakdown count berubah/menetap menuju solusi terbaik.
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(history['generation'], history['best_cost'], color='#1F4E79', linewidth=2.2, label='Best Cost')
ax1.set_xlabel('Generasi')
ax1.set_ylabel('Best Cost (lebih kecil lebih baik)', color='#1F4E79')
ax1.tick_params(axis='y', labelcolor='#1F4E79')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.step(history['generation'], history['best_breakdown_count'], where='post', color='#C0392B', linewidth=1.8, label='Breakdown Count')
ax2.set_ylabel('Breakdown Count', color='#C0392B')
ax2.tick_params(axis='y', labelcolor='#C0392B')

plt.title('Konvergensi Awal GA: Generasi vs Fitness/Cost')
fig.tight_layout()
fig.savefig(PLOTS / 'tahap3_ga_convergence.png', dpi=180)
plt.close(fig)

# Plot fitness for direct outline wording.
plt.figure(figsize=(9, 5))
plt.plot(history['generation'], history['best_fitness'], color='#148F77', linewidth=2.2, label='Best Fitness')
plt.plot(history['generation'], history['generation_mean_fitness'], color='#95A5A6', linewidth=1.5, linestyle='--', label='Mean Fitness per Generasi')
plt.xlabel('Generasi')
plt.ylabel('Fitness (lebih besar lebih baik)')
plt.title('Konvergensi Fitness GA Selama 60 Generasi')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / 'tahap3_ga_fitness_curve.png', dpi=180)
plt.close()

print('Ukuran input ML:', ml_input[FEATURES].shape)
print('Ukuran output ML:', predicted_rul.shape)
print('Skor terbaik GA:', best_summary['best_score'])
print('Urutan 15 mesin pertama:', best_summary['best_chromosome_machine_code_first_15'])
print('Output Tahap 3 tersimpan di', DATAOUT, 'dan', PLOTS)
