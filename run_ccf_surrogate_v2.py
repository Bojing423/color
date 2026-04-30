"""
任务二（修复版）：CCF Circular-Shift Surrogate 检验
=====================================================
使用 std-based 归一化（与用户模板一致）：
  ccf / (std(x) * std(y) * len(x))
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.signal import correlate
import os, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/bojingh/cognitive_color/paper_supplementary/task2_ccf_surrogate'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TASKS = {'b1': 'WM', 'b2': 'SR', 'b3': 'LI', 'b4': 'EF', 'b5': 'VA'}
EYE_METRICS = [
    'Pupil diameter left smoothed', 'Pupil diameter right smoothed', 'Pupil diameter average',
    'fixation_count', 'fixation_duration_sum', 'saccade_count', 'saccade_speed',
    'saccade_peak_speed', 'blink_count', 'blink_rate', 'IPA', 'LHIPA'
]
PARTICIPANTS = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,25,26,27,28,29,
                31,32,33,34,35,36,37,38,39,40,41,42,44,45,47,48,50,51,52,53,55,57,58]
MAX_LAG = 300
N_PERM = 200
MIN_SHIFT_FRAC = 0.05  # at least 5% of signal length
FS = 20

CHANNEL_TO_REGION = {
    1:'FPA',2:'FPA',3:'FPA',4:'FPA',7:'FPA',8:'FPA',9:'FPA',13:'FPA',14:'FPA',
    5:'DLPFC',6:'DLPFC',10:'DLPFC',11:'DLPFC',12:'DLPFC',15:'DLPFC',19:'DLPFC',
    20:'DLPFC',21:'DLPFC',26:'DLPFC',27:'DLPFC',28:'DLPFC',29:'DLPFC',32:'DLPFC',36:'DLPFC',
    16:'Broca',17:'Broca',18:'Broca',22:'Broca',23:'Broca',24:'Broca',25:'Broca',30:'Broca',31:'Broca',37:'Broca',
    33:'FEF',34:'FEF',35:'FEF',41:'FEF',42:'FEF',43:'FEF',44:'FEF',
    38:'PreM',39:'PreM',40:'PreM',45:'PreM',46:'PreM',47:'PreM',
    49:'PreM',50:'PreM',51:'PreM',52:'PreM',53:'PreM',54:'PreM',
    55:'PreM',56:'PreM',57:'PreM',62:'PreM',63:'PreM',64:'PreM',65:'PreM',73:'PreM',74:'PreM',75:'PreM',
    48:'TC',58:'TC',59:'TC',68:'TC',69:'TC',79:'TC',89:'TC',
    60:'SSC',67:'SSC',70:'SSC',71:'SSC',77:'SSC',78:'SSC',81:'SSC',82:'SSC',87:'SSC',88:'SSC',
    61:'PMC',66:'PMC',72:'PMC',76:'PMC',83:'PMC',84:'PMC',85:'PMC',86:'PMC',
    90:'OC',91:'OC',92:'OC',93:'OC',94:'OC',95:'OC',96:'OC',97:'OC',98:'OC',99:'OC',
    100:'OC',101:'OC',102:'OC',103:'OC',104:'OC',105:'OC',106:'OC'
}
ROIS = sorted(set(CHANNEL_TO_REGION.values()))

ROI_CHANNELS = {}
for ch, roi in CHANNEL_TO_REGION.items():
    ROI_CHANNELS.setdefault(roi, []).append(ch)

BASE_EYE = "/home/bojingh/cognitive_color/dataset/down"
BASE_OXY = "/home/bojingh/cognitive_color/dataset/fnirs_data"

def rank_center(signal):
    """Rank transform and center, returns numpy array."""
    r = rankdata(signal)
    return r - r.mean()

def compute_ccf_peak(x, y, max_lag=300):
    """Compute CCF with std normalization. Returns (|r_peak|, peak_lag_frames)."""
    n = len(x)
    if n < 2 * max_lag + 1:
        return None, None

    # Rank transform
    rx = rank_center(x)
    ry = rank_center(y)

    # Cross-correlation
    ccf = correlate(rx, ry, mode='full')
    center = n - 1

    # Extract lag range
    lo = center - max_lag
    hi = center + max_lag + 1
    if lo < 0 or hi > len(ccf):
        return None, None

    ccf_seg = ccf[lo:hi]

    # Normalize by std product × n → proper correlation scale
    sx, sy = np.std(rx), np.std(ry)
    if sx == 0 or sy == 0:
        return None, None
    ccf_norm = ccf_seg / (sx * sy * n)

    lags = np.arange(-max_lag, max_lag + 1)
    peak_idx = np.argmax(np.abs(ccf_norm))
    peak_r = np.abs(ccf_norm[peak_idx])
    peak_lag = lags[peak_idx]
    return peak_r, peak_lag

def circular_shift(signal, min_shift_frac=0.05):
    n = len(signal)
    min_shift = max(1, int(n * min_shift_frac))
    if n <= 2 * min_shift:
        return signal
    shift = np.random.randint(min_shift, n - min_shift)
    return np.roll(signal, shift)

# === Resume support ===
resume_file = os.path.join(OUTPUT_DIR, 'surrogate_results_v2.csv')
done_pairs = set()
results = []
if os.path.exists(resume_file):
    df_done = pd.read_csv(resume_file)
    results = df_done.to_dict('records')
    done_pairs = set(zip(df_done['participant'], df_done['task'], df_done['ROI'], df_done['eye_metric']))
    print(f"Resume: {len(done_pairs)} pairs done")

print(f"CCF Surrogate Test (std-normalized): {len(PARTICIPANTS)} ppts × 5 tasks × {len(ROIS)} ROIs × {len(EYE_METRICS)} metrics × {N_PERM} perms")

for pid in PARTICIPANTS:
    for block, task_name in TASKS.items():
        eye_file = f"{BASE_EYE}/{block}/{pid}/{pid}_eye.xlsx"
        oxy_file = f"{BASE_OXY}/{block}/{pid}/{pid}_oxy.xlsx"
        if not os.path.exists(eye_file) or not os.path.exists(oxy_file):
            continue
        try:
            eye_df = pd.read_excel(eye_file)
        except:
            continue
        try:
            oxy_sheets = pd.read_excel(oxy_file, sheet_name=None, header=None)
        except:
            continue

        # Build ROI fNIRS signals
        roi_signals = {}
        for roi_name, channels in ROI_CHANNELS.items():
            ch_arrays = []
            for sheet_data in oxy_sheets.values():
                sheet_data.columns = [str(c) for c in sheet_data.columns]
                for ch in channels:
                    if str(ch) in sheet_data.columns:
                        s = pd.to_numeric(sheet_data[str(ch)], errors='coerce').values
                        s = np.nan_to_num(s, nan=0.0)
                        ch_arrays.append(s)
            if ch_arrays:
                min_len = min(len(a) for a in ch_arrays)
                stacked = np.column_stack([a[:min_len] for a in ch_arrays])
                roi_signals[roi_name] = np.mean(stacked, axis=1)

        if not roi_signals:
            continue

        for eye_col in EYE_METRICS:
            if eye_col not in eye_df.columns:
                continue
            eye_raw = pd.to_numeric(eye_df[eye_col], errors='coerce').values
            eye_raw = np.nan_to_num(eye_raw, nan=0.0)
            if len(np.unique(eye_raw)) <= 1:
                continue

            for roi_name in ROIS:
                pair_key = (pid, task_name, roi_name, eye_col)
                if pair_key in done_pairs:
                    continue

                oxy_raw = roi_signals.get(roi_name)
                if oxy_raw is None:
                    continue
                if len(np.unique(oxy_raw)) <= 1:
                    continue

                # Align lengths
                ml = min(len(eye_raw), len(oxy_raw))
                x = eye_raw[:ml]
                y = oxy_raw[:ml]

                if len(x) < 2 * MAX_LAG + 1:
                    continue

                # Observed
                obs_r, obs_lag = compute_ccf_peak(x, y, MAX_LAG)
                if obs_r is None:
                    continue

                # Surrogates
                null_rs = []
                for _ in range(N_PERM):
                    x_shifted = circular_shift(x, MIN_SHIFT_FRAC)
                    null_r, _ = compute_ccf_peak(x_shifted, y, MAX_LAG)
                    if null_r is not None:
                        null_rs.append(null_r)

                if not null_rs:
                    continue

                null_rs = np.array(null_rs)
                p_val = (np.sum(null_rs >= obs_r) + 1) / (len(null_rs) + 1)

                results.append({
                    'participant': pid, 'task': task_name, 'ROI': roi_name,
                    'eye_metric': eye_col,
                    'observed_r': obs_r,
                    'observed_lag_s': obs_lag / FS,
                    'null_r_median': float(np.median(null_rs)),
                    'null_r_95': float(np.percentile(null_rs, 95)),
                    'p_val': p_val,
                    'significant': p_val < 0.05,
                })

        # Save after each participant-task
        if results and len(results) % 200 < 50:
            pd.DataFrame(results).to_csv(resume_file, index=False)

    pd.DataFrame(results).to_csv(resume_file, index=False)
    print(f"  Ppt {pid}: total {len(results)} results")

# === Aggregation ===
print(f"\n{'='*70}")
print("汇总")
print(f"{'='*70}")

df = pd.DataFrame(results)
print(f"总记录数: {len(df)}")
print(f"任务分布: {df['task'].value_counts().to_dict()}")

print(f"\n--- 每类任务显著配对比例 (p < 0.05) ---")
task_sig = df.groupby('task').agg(
    total=('p_val', 'count'), significant=('significant', 'sum')
).reset_index()
task_sig['proportion'] = task_sig['significant'] / task_sig['total']
for _, row in task_sig.iterrows():
    print(f"  {row['task']}: {row['significant']:.0f}/{row['total']} = {row['proportion']:.4f} ({row['proportion']*100:.1f}%)")

task_sig.to_csv(os.path.join(OUTPUT_DIR, 'task_significance_proportion.csv'), index=False)
df.to_csv(os.path.join(OUTPUT_DIR, 'surrogate_results_v2.csv'), index=False)

# Figure data
fig_obs = df[['task', 'observed_r']].copy()
fig_obs['type'] = 'observed'
fig_null = df[['task', 'null_r_median']].rename(columns={'null_r_median': 'observed_r'})
fig_null['type'] = 'surrogate_median'
fig_combined = pd.concat([fig_obs, fig_null])
fig_combined.to_csv(os.path.join(OUTPUT_DIR, 'figure_data_obs_vs_surrogate.csv'), index=False)

# Per-task stats for observed vs null
print(f"\n--- Observed vs Surrogate |r_peak| by task ---")
for task in ['WM', 'SR', 'LI', 'EF', 'VA']:
    tdf = df[df['task'] == task]
    if len(tdf) == 0:
        print(f"  {task}: no data")
        continue
    print(f"  {task}: obs_r = {tdf['observed_r'].median():.4f} [{tdf['observed_r'].quantile(.25):.4f}, {tdf['observed_r'].quantile(.75):.4f}], "
          f"null_r_median = {tdf['null_r_median'].median():.4f}, null_r_95 = {tdf['null_r_95'].median():.4f}")

print(f"\nSaved to {OUTPUT_DIR}/")
print("Done!")
