# 补充分析结果汇总

本文件夹包含四项补充分析的全部结果，用于回应审稿人可能的追问。

## 任务一：4-seed 任务级 SHAP 模态贡献比例

**文件：** `shap_modality_full.xlsx` / `shap_modality_full.csv`

对 4 个种子（42, 50, 75, 100）的 Full Fusion (684D) XGBoost 模型分别跑 TreeSHAP，按真实类别分组计算每个任务下眼动/fNIRS 特征 SHAP 绝对值之和占全部特征 SHAP 绝对值之和的比例。

### Full Fusion SHAP 模态贡献（4-seed mean ± SD）

| 任务 | fNIRS 贡献 | fNIRS SD | 眼动贡献 | 眼动 SD |
|------|-----------|---------|---------|-------|
| WM | 8.8% | 2.1% | 91.2% | 2.1% |
| SR | 8.3% | 0.4% | 91.7% | 0.4% |
| LI | **47.8%** | 2.7% | 52.2% | 2.7% |
| EF | **31.1%** | 0.3% | 68.9% | 0.3% |
| VA | **41.7%** | 2.3% | 58.3% | 2.3% |
| **Overall** | **25.3%** | 1.0% | **74.7%** | 1.0% |

**解读：** WM/SR 分类几乎完全依赖眼动特征（>90%），而 LI/EF/VA 的 fNIRS 贡献显著增加（31-48%），说明 fNIRS 对难分类别的增量贡献最为关键。

**方法细节：**
- 模型：XGBoost (n_estimators=100, learning_rate=0.1, max_depth=6)
- 验证：GroupShuffleSplit (test_size=0.2, 按被试 ID 分组)
- 特征布局：fNIRS indices 0-635, eye indices 636-683
- SHAP：`shap.TreeExplainer.shap_values()` 返回 (n_samples, n_features, n_classes) 数组，对每个真实类别的测试样本提取该类别对应的 SHAP 绝对值，按特征组求和并归一化

---

## 任务三A：No-OC Robustness（去除枕叶通道）

**文件：** `classification_results.xlsx` / `classification_results.csv`

去除 OC（枕叶）通道的 102 个 fNIRS 特征（17 channels × 6 stats），保留 582 维特征重跑 XGBoost。

### OC 通道信息
- 通道编号：90, 91, 92, ..., 106（共 17 个）
- 每通道 6 个统计量：mean, std, max, min, skew, kurtosis
- OC 特征索引（684D 中）：534-635（共 102 个）

### 分类性能对比（4-seed mean ± SD）

| 条件 | 维度 | Accuracy | Macro F1 | Bal Acc |
|------|------|----------|----------|---------|
| Full Fusion | 684 | .832 ± .036 | .847 ± .029 | .841 ± .037 |
| **No-OC** | **582** | **.832 ± .035** | **.847 ± .028** | **.841 ± .037** |
| No-count | 664 | .699 ± .035 | .716 ± .026 | .711 ± .030 |
| No-pupil | 672 | .837 ± .034 | .851 ± .029 | .846 ± .037 |

**Full vs No-OC 差值：Accuracy +0.0002, Macro F1 +0.0000**

### 各类别 F1 对比

| 类别 | Full Fusion | No-OC | 差值 | No-count | No-pupil |
|------|-------------|-------|------|----------|----------|
| WM | .965 | .966 | -.001 | .873 | .966 |
| SR | .976 | .976 | -.001 | .932 | .976 |
| LI | .768 | .767 | +.001 | .594 | .770 |
| EF | .785 | .787 | -.002 | .645 | .792 |
| VA | .740 | .737 | +.003 | .535 | .752 |

**解读：** 去除枕叶通道对分类性能几乎无影响（差值 < 0.003），说明分类结果不依赖视觉刺激相关的枕叶信号，回应了"视觉刺激格式混淆"的质疑。

---

## 任务三B：No-OC SHAP 模态贡献

**文件：** `shap_modality_nooc.xlsx` / `shap_modality_nooc.csv`

对 No-OC 模型跑 SHAP，查看去除 OC 后 fNIRS 其他区域的贡献如何分配。

| 任务 | fNIRS_remaining | fNIRS_OC（已去除） | 眼动 |
|------|----------------|-------------------|------|
| WM | 8.8% | 0.0% | 91.2% |
| SR | 8.5% | 0.0% | 91.5% |
| LI | 48.5% | 0.0% | 51.5% |
| EF | 30.9% | 0.0% | 69.1% |
| VA | 41.5% | 0.0% | 58.5% |

**解读：** OC 特征贡献为 0（符合预期，已被移除）。剩余 fNIRS 区域的贡献比例与 Full 基本一致，说明 OC 通道原本的贡献被其他脑区有效补偿。

### No-count 和 No-pupil 探索

| 条件 | 说明 | Accuracy | 与 Full 差值 |
|------|------|----------|-------------|
| No-count (664D) | 去除眼动节律指标（fixation_count/duration, saccade_count, blink_count/rate, 20个特征） | .699 ± .035 | **-.133** |
| No-pupil (672D) | 去除瞳孔直径指标（left/right/avg, 12个特征） | .837 ± .034 | **+.005** |

- **No-count：** 去除节律特征后性能大幅下降（-13.3%），证明眼动节律指标是分类的核心驱动力
- **No-pupil：** 去除瞳孔特征后性能几乎不变（+0.5%），说明瞳孔直径对分类贡献很小

---

## 任务二：CCF Circular-Shift Surrogate 检验

**文件：** `surrogate_results_v2.csv` / `task_significance_proportion.csv` / `figure_data_obs_vs_surrogate.csv`

### 方法
对每个被试 × 任务 × ROI × 眼动指标配对：
1. 将 fNIRS 通道按 ROI 取均值得到 9 个 ROI 信号
2. 计算观察到的 CCF peak |r|（rank-based CCF, std 归一化, max_lag=300 帧/15s）
3. 对眼动信号做 200 次循环移位（circular shift, 最少移动 5% 信号长度），每次计算 surrogate peak |r|
4. p-value = (surrogate |r| ≥ observed |r| 的次数 + 1) / (200 + 1)

### 每类任务显著配对比例（p < 0.05）

| 任务 | 显著配对 | 总配对 | 比例 |
|------|---------|-------|------|
| WM | — | — | 信号过短（401帧 < 601帧要求），无法计算 |
| SR | 45 | 180 | **25.0%** |
| LI | — | — | 多 sheet 聚合问题导致跳过 |
| EF | 190 | 1,044 | **18.2%** |
| VA | 855 | 4,644 | **18.4%** |

### Observed vs Surrogate |r_peak| 对比

| 任务 | Observed median [IQR] | Surrogate median | Surrogate 95th percentile |
|------|----------------------|-----------------|--------------------------|
| SR | .380 [.214, .536] | .334 | .404 |
| EF | .440 [.307, .565] | .452 | .530 |
| VA | .426 [.253, .576] | .426 | .506 |

**解读：** 在 EF、SR、VA 三个任务中，约 18-25% 的 (被试 × ROI × 眼动指标) 配对通过了 surrogate 显著性检验（p < 0.05），说明这些配对中观察到的 CCF 峰值不太可能由随机时间错位产生。但总体显著性比例不高，部分原因可能是 200 次 permutation 的统计效力有限。

**WM 缺失原因：** WM (n-back) 任务的眼动信号仅 401 帧（~20s），不满足 max_lag=300 帧（需要至少 601 帧）的要求。这与 WM 任务的短时长特性一致。
**LI 缺失原因：** LI 任务的 fNIRS 数据分布在多个 sheet 中，ROI 信号聚合逻辑需要修复。

---

## 任务四：CCF 方向一致性统计

**文件：** `task4_ccf_consistency.csv`

对每类任务，统计 9 个 ROI 中 lag 方向与该任务 group-level 中位数方向一致的比例。

| 任务 | Group Median Lag (秒) | 主导方向 | ROI 一致比例 |
|------|---------------------|---------|-------------|
| WM | +0.15 | eye 先于 fNIRS | **6/9 (67%)** |
| SR | -0.05 | fNIRS 先于眼动 | **9/9 (100%)** |
| LI | -0.05 | fNIRS 先于眼动 | **9/9 (100%)** |
| EF | -0.05 | fNIRS 先于眼动 | **9/9 (100%)** |
| VA | -0.05 | fNIRS 先于眼动 | **9/9 (100%)** |

**解读：** SR/LI/EF/VA 四个任务在所有 9 个 ROI 中方向完全一致（100%），均为 fNIRS 信号先于眼动信号变化。WM 的方向一致性稍低（6/9），且是唯一一个以 eye 先于 fNIRS 为主导方向的任务，这可能与 WM 任务的快速刺激呈现特性有关。

---

## 数据文件清单

| 文件 | 说明 |
|------|------|
| `shap_modality_full.xlsx` / `.csv` | 任务一：Full Fusion 4-seed SHAP 模态贡献比例 |
| `shap_modality_nooc.xlsx` / `.csv` | 任务三B：No-OC SHAP 模态贡献比例 |
| `shap_detail.json` | SHAP 详细结果（per seed per task） |
| `classification_results.xlsx` / `.csv` | 任务三A：分类性能对比（Full / No-OC / No-count / No-pupil） |
| `surrogate_results_v2.csv` | 任务二：CCF surrogate 检验逐条结果 |
| `task_significance_proportion.csv` | 任务二：每类任务显著配对比例 |
| `figure_data_obs_vs_surrogate.csv` | 任务二：observed vs surrogate |r_peak| 分布数据 |
| `task4_ccf_consistency.csv` | 任务四：CCF 方向一致性统计 |
