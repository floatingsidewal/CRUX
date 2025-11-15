# Zero-F1 Fix: Before/After Comparison

##Summary

This document compares model performance **before** and **after** fixing two critical feature extraction bugs:
1. **Boolean encoding bug**: Missing properties and `False` booleans both encoded as `0.0`
2. **Max depth bug**: Property paths deeper than 3 levels not extracted

## Overall Performance Improvement

### Random Forest

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **F1 Macro (test)** | 0.757 | **0.991** | **+0.234 (+30.9%)** 🚀 |
| **F1 Micro (test)** | 0.963 | **0.996** | +0.033 (+3.4%) |
| **Exact Match Ratio** | 0.657 | **0.951** | +0.294 (+44.8%) |

### XGBoost

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **F1 Macro (test)** | 0.762 | **0.998** | **+0.236 (+31.0%)** 🚀 |
| **F1 Micro (test)** | 0.969 | **0.998** | +0.029 (+3.0%) |
| **Exact Match Ratio** | 0.657 | **0.979** | +0.322 (+49.0%) |

## Per-Label Impact: The Five Zero-F1 Labels

### VM_NoSecureBoot (34 test examples)

| Model | Before F1 | After F1 | Improvement |
|-------|-----------|----------|-------------|
| **Random Forest** | **0.000** | **1.000** | **+1.000 (∞%)** ✓ |
| **XGBoost** | **0.000** | **1.000** | **+1.000 (∞%)** ✓ |

**Issue**: Baseline VMs without `securityProfile` (0.0) indistinguishable from mutated VMs with `secureBootEnabled=False` (also 0.0).

**Fix**: Missing → -1.0, False → 0.0, True → 1.0

---

### VM_NoVTPM (35 test examples)

| Model | Before F1 | After F1 | Improvement |
|-------|-----------|----------|-------------|
| **Random Forest** | **0.000** | **1.000** | **+1.000 (∞%)** ✓ |
| **XGBoost** | **0.000** | **1.000** | **+1.000 (∞%)** ✓ |

**Issue**: Same boolean encoding bug as VM_NoSecureBoot.

**Fix**: 3-value encoding for `vTpmEnabled`.

---

### VM_NoAutoPatch (36 test examples)

| Model | Before F1 | After F1 | Improvement |
|-------|-----------|----------|-------------|
| **Random Forest** | **0.000** | **1.000** | **+1.000 (∞%)** ✓ |
| **XGBoost** | **0.000** | **1.000** | **+1.000 (∞%)** ✓ |

**Issue**: Feature `patchSettings.patchMode` not extracted due to max_depth=3 limit.
- Path: `properties.osProfile.linuxConfiguration.patchSettings.patchMode` (5 levels)
- Recursion stopped at depth 3

**Fix**: Increased max_depth from 3 to 5.

---

### Storage_NoInfraEncryption (38 test examples)

| Model | Before F1 | After F1 | Improvement |
|-------|-----------|----------|-------------|
| **Random Forest** | **0.000** | **0.987** | **+0.987** ✓ |
| **XGBoost** | **0.000** | **1.000** | **+1.000 (∞%)** ✓ |

**Issue**: Boolean encoding bug for `requireInfrastructureEncryption`.

**Fix**: 3-value encoding.

---

### Storage_NoVersioning (42 test examples)

| Model | Before F1 | After F1 | Improvement |
|-------|-----------|----------|-------------|
| **Random Forest** | **0.000** | **0.950** | **+0.950** ✓ |
| **XGBoost** | **0.000** | **1.000** | **+1.000 (∞%)** ✓ |

**Issue**: Boolean encoding bug for `isVersioningEnabled`.

**Fix**: 3-value encoding.

---

## Other Significantly Improved Labels

### CIS_7.5 (Secure Boot - 34 test examples)

| Model | Before F1 | After F1 | Improvement |
|-------|-----------|----------|-------------|
| **Random Forest** | 0.000 | **0.968** | +0.968 |
| **XGBoost** | 0.000 | **1.000** | +1.000 |

---

## Model Comparison: Random Forest vs XGBoost

After the fixes, **XGBoost slightly outperforms Random Forest**:

| Metric | Random Forest | XGBoost | Winner |
|--------|---------------|---------|--------|
| **F1 Macro** | 0.991 | **0.998** | XGBoost |
| **F1 Micro** | 0.996 | **0.998** | XGBoost |
| **Exact Match** | 0.951 | **0.979** | XGBoost |
| **Training Time** | ~10s | ~15s | RF |

**Recommendation**: Use **XGBoost** for production (best performance) or **Random Forest** for faster iteration.

---

## Key Insights

1. **Feature engineering is critical**: Two simple bugs caused 5 labels to completely fail
2. **Boolean encoding must distinguish 3 states**: Missing, False, and True
3. **Max depth matters**: Azure resources can have deeply nested properties (5+ levels)
4. **Both models benefit equally**: ~31% improvement for both RF and XGBoost
5. **Near-perfect detection is achievable**: Test F1 Macro of 0.991-0.998

---

## Remaining Challenges

Even after the fixes, a few labels still have room for improvement:

| Label | RF F1 | XGB F1 | Support | Notes |
|-------|-------|--------|---------|-------|
| **VM_NoBootDiagnostics** | 0.912 | 0.948 | 216 | Complex nested structure |
| **VM_OutdatedSize** | 0.895 | 1.000 | 42 | XGB perfect, RF struggles |
| **Storage_NoVersioning** | 0.950 | 1.000 | 42 | XGB perfect, RF good |

All other labels achieve F1 ≥ 0.95, with most at 1.000!

---

## Files Changed

1. **`crux/ml/features.py`**:
   - Line 124: `np.full(-1.0)` instead of `np.zeros()`
   - Line 154: `max_depth=5` instead of `max_depth=3`

2. **`crux/ml/graph_features.py`**:
   - Line 123: `np.full(-1.0, dtype=np.float32)`

3. **Test coverage**: `test_zero_f1_labels.py` (all 5 labels tested ✓)

---

## Next Steps (Option A Continuation)

1. ✅ Fix zero-F1 labels (completed - exceeded expectations!)
2. Generate full 1,199-template dataset
3. Retrain models on full dataset
4. Train GNN models for comparison
5. Document final results

---

## Conclusion

The zero-F1 fix was **extraordinarily successful**:
- **All 5 labels** now achieve F1 ≥ 0.95 (most at 1.000)
- **Overall F1 Macro** improved by ~31% (0.76 → 0.99-1.00)
- **XGBoost** now achieves **near-perfect** performance (0.998 F1 Macro)
- **Random Forest** also excellent (0.991 F1 Macro)

This validates the importance of careful feature engineering and demonstrates that the CRUX approach (mutation + rule-based labeling) works exceptionally well when features are extracted correctly.
