"""
任务一 + 三A + 三B + No-count/No-pupil
======================================
1. 4-seed 任务级 SHAP 模态贡献比例 (眼动 vs fNIRS)
2. No-OC Robustness (去除枕叶通道重跑 XGBoost)
3. No-OC SHAP 模态贡献
4. No-count / No-pupil 探索性分析

特征布局 (684D):
  fNIRS: indices 0-635 (106 channels × 6 stats)
  Eye:   indices 636-683 (12 metrics × 4 stats)

OC channels: 90-106 → fNIRS indices (89*6)=534 to (105*6+5)=635 → 102 features
Rhythm metrics: [0,1,2,8,9] × 4 stats = 20 eye features
Pupil metrics: [5,6,7] × 4 stats = 12 eye features
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from xgboost import XGBClassifier
import os, json, warnings
os.environ['NUMBA_DISABLE_JIT'] = '1'
import shap
warnings.filterwarnings('ignore')

CACHE = '/home/bojingh/cognitive_color/five_class_results/final_results_for_paper/unified_results/unified_data_cache.npz'
OUTPUT_DIR = '/home/bojingh/cognitive_color/paper_supplementary/task1_shap_task3_nooc'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 50, 75, 100]
TASK_NAMES = ['WM', 'SR', 'LI', 'EF', 'VA']

# ============================================================
# 加载数据
# ============================================================
print("加载数据...")
d = np.load(CACHE, allow_pickle=True)
fnirs = d['fnirs']  # (N, 636)
eye = d['eye']      # (N, 48)
labels = d['labels']
ids = d['ids']

mask = np.isnan(fnirs).any(axis=1) | np.isinf(fnirs).any(axis=1) | np.isnan(eye).any(axis=1) | np.isinf(eye).any(axis=1)
if mask.sum() > 0:
    fnirs, eye, labels, ids = fnirs[~mask], eye[~mask], labels[~mask], ids[~mask]
    print(f"  移除 {mask.sum()} 个异常样本")

X_full = np.concatenate([fnirs, eye], axis=1)  # (N, 684)
N = len(labels)
print(f"  样本数: {N}, 特征: {X_full.shape[1]}")

# ============================================================
# 特征索引定义
# ============================================================
# fNIRS: 106 channels, 6 stats each → 636 features (indices 0-635)
# Channel N (1-based) → indices (N-1)*6 to (N-1)*6+5

OC_CHANNELS = list(range(90, 107))  # 17 channels
oc_feat_idx = []
for ch in OC_CHANNELS:
    start = (ch - 1) * 6
    oc_feat_idx.extend(range(start, start + 6))
# oc_feat_idx: 534-635 (102 features)

# Eye: 12 metrics × 4 stats = 48 features (indices 636-683)
# Metrics: 0=FixCnt, 1=FixDur, 2=SaccCnt, 3=SaccSpd, 4=SaccPeak, 5=PupilL, 6=PupilR, 7=PupilAvg, 8=BlinkCnt, 9=BlinkRate, 10=IPA, 11=LHIPA
# Stats: mean(0-11), std(12-23), max(24-35), min(36-47)

RHYTHM_METRICS = [0, 1, 2, 8, 9]
PUPIL_METRICS = [5, 6, 7]

rhythm_eye_idx = [636 + so + m for so in [0, 12, 24, 36] for m in RHYTHM_METRICS]  # 20 features
pupil_eye_idx = [636 + so + m for so in [0, 12, 24, 36] for m in PUPIL_METRICS]    # 12 features

fnirs_idx = list(range(0, 636))
eye_idx = list(range(636, 684))
eye_nr_idx = [i for i in eye_idx if i not in rhythm_eye_idx]  # 28 non-rhythm eye

# No-OC: remove OC fNIRS features
no_oc_idx = [i for i in range(684) if i not in oc_feat_idx]
# No-count: remove rhythm eye features
no_count_idx = [i for i in range(684) if i not in rhythm_eye_idx]
# No-pupil: remove pupil eye features
no_pupil_idx = [i for i in range(684) if i not in pupil_eye_idx]

print(f"  Full Fusion: 684D")
print(f"  No-OC: {len(no_oc_idx)}D (移除 {len(oc_feat_idx)} 个OC特征)")
print(f"  No-count: {len(no_count_idx)}D (移除 {len(rhythm_eye_idx)} 个rhythm特征)")
print(f"  No-pupil: {len(no_pupil_idx)}D (移除 {len(pupil_eye_idx)} 个pupil特征)")

# ============================================================
# 模型训练和评估函数
# ============================================================
def get_xgb(seed):
    return XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                         random_state=seed, n_jobs=-1,
                         use_label_encoder=False, eval_metric='mlogloss')

def train_and_eval(X, y, groups, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    model = get_xgb(seed)
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])
    y_test = y[test_idx]
    per_f1 = f1_score(y_test, y_pred, average=None)
    return {
        'model': model, 'X_test': X[test_idx], 'y_test': y_test,
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'balanced_acc': balanced_accuracy_score(y_test, y_pred),
        'per_class_f1': {TASK_NAMES[i]: float(per_f1[i]) for i in range(5)},
    }

def compute_shap_modality(model, X_test, y_test, feat_groups, label=""):
    """计算每个任务类别的模态SHAP贡献比例"""
    print(f"  计算 SHAP: {label}")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)

    # 处理不同 shap 版本的输出格式
    if isinstance(sv, list):
        # list of (n_samples, n_features) per class
        sv = np.stack(sv, axis=-1)  # → (n_samples, n_features, n_classes)
    elif hasattr(sv, 'values'):
        sv = sv.values  # Explanation object → numpy
    # 现在 sv 应该是 (n_samples, n_features, n_classes)

    task_results = {}
    for task_id, task_name in enumerate(TASK_NAMES):
        mask = (y_test == task_id)
        if mask.sum() == 0:
            continue
        abs_sv = np.abs(sv[mask, :, task_id])  # (n_samples, n_features)
        total = abs_sv.sum()
        if total == 0:
            continue

        group_contrib = {}
        for gname, gidx in feat_groups.items():
            group_contrib[gname] = float(abs_sv[:, gidx].sum() / total)
        task_results[task_name] = group_contrib

    # 整体 (所有测试样本)
    all_abs = np.abs(sv)  # (n_samples, n_features, n_classes)
    total_all = all_abs.sum()
    overall = {}
    if total_all > 0:
        for gname, gidx in feat_groups.items():
            overall[gname] = float(all_abs[:, gidx, :].sum() / total_all)

    return task_results, overall

# ============================================================
# 运行 4 个条件
# ============================================================
conditions = {
    'Full Fusion (684D)': {
        'idx': list(range(684)),
        'feat_groups': {'fNIRS': fnirs_idx, 'eye': eye_idx},
        'X': X_full,
        'shap_groups_for_nooc': None,
    },
    'No-OC (582D)': {
        'idx': no_oc_idx,
        'feat_groups': {
            'fNIRS_noOC': [i for i in fnirs_idx if i not in oc_feat_idx],
            'fNIRS_OC': oc_feat_idx,
            'eye': eye_idx,
        },
        'X': X_full[:, no_oc_idx],
        'shap_groups_for_nooc': {
            'fNIRS_remaining': list(range(len([i for i in fnirs_idx if i not in oc_feat_idx]))),
            'eye': list(range(len([i for i in fnirs_idx if i not in oc_feat_idx]),
                             len([i for i in fnirs_idx if i not in oc_feat_idx]) + len(eye_idx))),
        },
    },
    'No-count (664D)': {
        'idx': no_count_idx,
        'feat_groups': {'fNIRS': fnirs_idx, 'eye_nonrhythm': eye_nr_idx},
        'X': X_full[:, no_count_idx],
    },
    'No-pupil (672D)': {
        'idx': no_pupil_idx,
        'feat_groups': {
            'fNIRS': fnirs_idx,
            'eye_nonpupil': [i for i in eye_idx if i not in pupil_eye_idx],
        },
        'X': X_full[:, no_pupil_idx],
    },
}

# 存储所有结果
all_shap_results = {}  # {condition: {seed: {task: {group: prop}}}}
all_class_results = []  # classification metrics

for cond_name, cond in conditions.items():
    print(f"\n{'='*70}")
    print(f"条件: {cond_name}")
    print(f"{'='*70}")

    X_cond = cond['X']
    feat_groups = cond['feat_groups']
    # Map feat_groups to local indices (relative to the condition's feature matrix)
    global_idx = cond['idx']
    idx_map = {g: [global_idx.index(i) for i in gidx if i in global_idx]
               for g, gidx in feat_groups.items()}

    cond_shap = {}
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        res = train_and_eval(X_cond, labels, ids, seed)
        print(f"    Acc={res['accuracy']:.4f}, F1={res['macro_f1']:.4f}")
        print(f"    Per-class: " + ", ".join(f"{t}={res['per_class_f1'][t]:.3f}" for t in TASK_NAMES))

        all_class_results.append({
            'condition': cond_name, 'seed': seed,
            'accuracy': res['accuracy'], 'macro_f1': res['macro_f1'],
            'balanced_acc': res['balanced_acc'],
            **{f'f1_{t}': res['per_class_f1'][t] for t in TASK_NAMES},
        })

        # SHAP only for Full Fusion and No-OC
        if cond_name in ['Full Fusion (684D)', 'No-OC (582D)']:
            task_shap, overall_shap = compute_shap_modality(
                res['model'], res['X_test'], res['y_test'],
                idx_map, label=f"{cond_name} seed {seed}")
            cond_shap[seed] = {'per_task': task_shap, 'overall': overall_shap}

            for t in TASK_NAMES:
                if t in task_shap:
                    parts = " ".join(f"{g}={v:.4f}" for g, v in task_shap[t].items())
                    print(f"    SHAP {t}: {parts}")

    if cond_shap:
        all_shap_results[cond_name] = cond_shap

# ============================================================
# 汇总输出
# ============================================================
print(f"\n\n{'='*80}")
print("分类性能汇总 (4-seed mean ± SD)")
print(f"{'='*80}")

cond_order = list(conditions.keys())
class_df = pd.DataFrame(all_class_results)

print(f"\n{'条件':<25s} {'Accuracy':>18s} {'Macro F1':>18s} {'Bal Acc':>18s}")
print("-" * 80)
for cond_name in cond_order:
    sub = class_df[class_df['condition'] == cond_name]
    acc = sub['accuracy']
    mf1 = sub['macro_f1']
    bacc = sub['balanced_acc']
    print(f"{cond_name:<25s} {acc.mean():.4f}±{acc.std(ddof=1):.3f}   "
          f"{mf1.mean():.4f}±{mf1.std(ddof=1):.3f}   "
          f"{bacc.mean():.4f}±{bacc.std(ddof=1):.3f}")

print(f"\n{'条件':<25s} {'F1_WM':>10s} {'F1_SR':>10s} {'F1_LI':>10s} {'F1_EF':>10s} {'F1_VA':>10s}")
print("-" * 75)
for cond_name in cond_order:
    sub = class_df[class_df['condition'] == cond_name]
    vals = {t: sub[f'f1_{t}'] for t in TASK_NAMES}
    print(f"{cond_name:<25s} " + " ".join(f"{vals[t].mean():.4f}±{vals[t].std(ddof=1):.3f}" for t in TASK_NAMES))

# Full vs No-OC 对比
print(f"\n{'='*80}")
print("Full vs No-OC 对比")
print(f"{'='*80}")
full = class_df[class_df['condition'] == 'Full Fusion (684D)']
nooc = class_df[class_df['condition'] == 'No-OC (582D)']
print(f"  Full:  Acc={full['accuracy'].mean():.4f}±{full['accuracy'].std(ddof=1):.3f}  F1={full['macro_f1'].mean():.4f}±{full['macro_f1'].std(ddof=1):.3f}")
print(f"  No-OC: Acc={nooc['accuracy'].mean():.4f}±{nooc['accuracy'].std(ddof=1):.3f}  F1={nooc['macro_f1'].mean():.4f}±{nooc['macro_f1'].std(ddof=1):.3f}")
print(f"  差值:  Acc={full['accuracy'].mean()-nooc['accuracy'].mean():+.4f}  F1={full['macro_f1'].mean()-nooc['macro_f1'].mean():+.4f}")
for t in TASK_NAMES:
    diff = full[f'f1_{t}'].mean() - nooc[f'f1_{t}'].mean()
    print(f"    {t}: Full={full[f'f1_{t}'].mean():.4f} vs No-OC={nooc[f'f1_{t}'].mean():.4f}  差值={diff:+.4f}")

# ============================================================
# 任务一输出：SHAP 模态贡献比例 Excel
# ============================================================
print(f"\n{'='*80}")
print("任务一：SHAP 模态贡献比例 (4-seed mean ± SD)")
print(f"{'='*80}")

for cond_name in ['Full Fusion (684D)', 'No-OC (582D)']:
    if cond_name not in all_shap_results:
        continue
    print(f"\n--- {cond_name} ---")

    # 收集 4 seeds 的结果
    shap_rows = []
    groups = list(all_shap_results[cond_name][SEEDS[0]]['per_task'][TASK_NAMES[0]].keys())

    # Per-task
    for t in TASK_NAMES:
        row = {'Task': t}
        for g in groups:
            vals = [all_shap_results[cond_name][s]['per_task'][t][g] for s in SEEDS if t in all_shap_results[cond_name][s]['per_task']]
            row[f'{g}_mean'] = np.mean(vals)
            row[f'{g}_sd'] = np.std(vals, ddof=1)
        shap_rows.append(row)

    # Overall
    row = {'Task': 'Overall'}
    for g in groups:
        vals = [all_shap_results[cond_name][s]['overall'][g] for s in SEEDS if g in all_shap_results[cond_name][s]['overall']]
        row[f'{g}_mean'] = np.mean(vals)
        row[f'{g}_sd'] = np.std(vals, ddof=1)
    shap_rows.append(row)

    shap_df = pd.DataFrame(shap_rows)
    print(shap_df.to_string(index=False))

    # Save
    tag = 'full' if 'Full' in cond_name else 'nooc'
    shap_df.to_csv(os.path.join(OUTPUT_DIR, f'shap_modality_{tag}.csv'), index=False)
    shap_df.to_excel(os.path.join(OUTPUT_DIR, f'shap_modality_{tag}.xlsx'), index=False)
    print(f"  已保存: shap_modality_{tag}.xlsx")

# 保存所有分类结果
class_df.to_csv(os.path.join(OUTPUT_DIR, 'classification_results.csv'), index=False)
class_df.to_excel(os.path.join(OUTPUT_DIR, 'classification_results.xlsx'), index=False)

# 保存 SHAP 详细 JSON
with open(os.path.join(OUTPUT_DIR, 'shap_detail.json'), 'w') as f:
    json.dump({cond: {str(s): v for s, v in seeds.items()} for cond, seeds in all_shap_results.items()}, f, indent=2)

print(f"\n结果保存到: {OUTPUT_DIR}/")
print("完成！")
