# Fix for Zero-F1 Labels: Boolean Feature Encoding Issue

## Problem Statement

Five labels consistently achieved F1=0 despite having 30-40+ training examples:
- `VM_NoSecureBoot` (34 test examples, F1=0)
- `VM_NoVTPM` (35 test examples, F1=0)
- `VM_NoAutoPatch` (36 test examples, F1=0)
- `Storage_NoInfraEncryption` (38 test examples, F1=0)
- `Storage_NoVersioning` (42 test examples, F1=0)

## Root Cause Analysis

### Investigation Process

1. **Verified mutations and rules work correctly** (`test_zero_f1_labels.py`):
   - All 5 mutations apply correctly to test resources
   - All 5 rules detect the mutations properly
   - ✓ Mutation/rule logic is correct

2. **Verified features are extracted** (dataset analysis):
   - All critical features present in feature set:
     - Feature 50: `prop_securityProfile.uefiSettings.secureBootEnabled`
     - Feature 51: `prop_securityProfile.uefiSettings.vTpmEnabled`
     - Feature 22: `prop_encryption.requireInfrastructureEncryption`
     - Feature 25: `prop_isVersioningEnabled`
   - ✓ Features are being extracted

3. **Discovered the encoding bug**:
   - **Baseline VMs**: 312/315 don't have `secureBootEnabled` property → encoded as **0.0**
   - **Mutated VMs**: `secureBootEnabled = False` → also encoded as **0.0**
   - **INDISTINGUISHABLE!** Model cannot learn to separate them.

### Technical Details

In `crux/ml/features.py` (and `graph_features.py`):

**Before Fix:**
```python
features = np.zeros(len(self.feature_names))  # Missing → 0.0
...
if isinstance(value, bool):
    result[path] = 1.0 if value else 0.0      # False → 0.0
```

**Encoding:**
- Missing property → `0.0` (default)
- `False` boolean → `0.0` (explicit)
- `True` boolean → `1.0`

**Result:** Missing and `False` are the same value!

### Real-World Example

**VM_NoSecureBoot mutation:**
- Sets `properties.securityProfile.uefiSettings.secureBootEnabled = False`
- Rule checks if this property equals `false`

**Dataset distribution:**
- 315 baseline VMs total
- 312 VMs (99%) don't have `securityProfile` at all
- 3 VMs (1%) have it set to `"[parameters('secureBoot')]"` (a parameter reference string)
- 175 mutated VMs have `secureBootEnabled = False`

**Feature values:**
- Baseline VMs without property: `0.0`
- Mutated VMs with `False`: `0.0`
- **Model sees no difference!**

## Solution

Use **3-value encoding** to distinguish all states:
- Missing property → `-1.0`
- `False` boolean → `0.0`
- `True` boolean → `1.0`

### Implementation

**After Fix:**
```python
# Initialize with -1.0 to distinguish missing properties from False booleans
features = np.full(len(self.feature_names), -1.0)
...
if isinstance(value, bool):
    result[path] = 1.0 if value else 0.0  # False → 0.0, True → 1.0
# Missing properties remain -1.0
```

## Changes Made

### Files Modified

1. **`crux/ml/features.py`**:
   - Line 124-125 (Boolean encoding fix):
     ```python
     # OLD: features = np.zeros(len(self.feature_names))
     # NEW: features = np.full(len(self.feature_names), -1.0)
     ```
   - Line 154 (Max depth fix):
     ```python
     # OLD: max_depth: int = 3
     # NEW: max_depth: int = 5
     ```

2. **`crux/ml/graph_features.py`** (Line 123-124):
   ```python
   # OLD: features = np.zeros(num_features, dtype=np.float32)
   # NEW: features = np.full(num_features, -1.0, dtype=np.float32)
   ```
   Note: Graph features already had no max_depth limit, so patchMode works correctly.

### Test Created

`test_zero_f1_labels.py`: Comprehensive test verifying:
- Mutations apply correctly
- Rules detect mutations properly
- All 5 zero-F1 labels work in isolation

## Expected Impact

### Before Fix
- VM_NoSecureBoot: F1=0 (34 examples)
- VM_NoVTPM: F1=0 (35 examples)
- VM_NoAutoPatch: F1=0 (36 examples)
- Storage_NoInfraEncryption: F1=0 (38 examples)
- Storage_NoVersioning: F1=0 (42 examples)

### After Fix (Expected)
- Models can now distinguish missing properties from `False` values
- Expected F1 improvement: **0.0 → 0.5-0.8** for these labels
- Overall F1 Macro improvement: **+0.08 to +0.12** (+10-15%)

## Verification

Run this to verify the fix:
```bash
# Test feature encoding
python test_zero_f1_labels.py

# Expected output:
# VM1 (True):    1.0
# VM2 (False):   0.0
# VM3 (Missing): -1.0
# ✓ All assertions passed!
```

## Second Issue Fixed: VM_NoAutoPatch Max Depth

**Problem:** `VM_NoAutoPatch` feature `patchSettings.patchMode` was **not extracted** at all.

**Root Cause:** Property path is too deep for `max_depth=3`:
- `properties.osProfile.linuxConfiguration.patchSettings.patchMode` = **5 levels**
- Default `max_depth=3` stopped at `linuxConfiguration`

**Fix:** Increased `max_depth` from 3 to 5 in `_extract_property_paths()`

**Verification:**
```python
# Before: Feature not extracted
# After: prop_osProfile.linuxConfiguration.patchSettings.patchMode extracted ✓
```

## Next Steps

1. ✅ Fix boolean encoding (completed)
2. ✅ Fix `VM_NoAutoPatch` max_depth issue (completed)
3. Retrain models on large-500 dataset with fixes
4. Compare before/after metrics
5. Document improvements in Milestone 4.2 (or 4.1.1)

## Lessons Learned

1. **Feature encoding matters**: Subtle bugs in feature engineering can completely prevent learning
2. **Test end-to-end**: Mutations and rules can work perfectly, but feature extraction can still fail
3. **Distinguish missing vs. false**: For boolean features, always use 3-value encoding
4. **Verify assumptions**: "Missing defaults to 0" seems harmless but breaks boolean features
