# Research Paper Comparison & Improvement Recommendations

## 📊 Current Setup vs Research Papers

### 1. **Data Source** ⚠️ CRITICAL DIFFERENCE

| Aspect | Imperial College (DLSTM) | Your Setup | Impact |
|--------|-------------------------|-----------|--------|
| **Data Type** | **Limit Order Book (LOB)** - 40 features (bid/ask prices & volumes at 10 levels) | **OHLCV** - 5 features | 🔴 **HIGH** - LOB contains much richer microstructure information |
| **Features** | 41 features (40 LOB + mid-price) | 5-20 features (OHLCV + technical indicators) | 🟡 **MEDIUM** - Technical indicators help but LOB is superior |
| **Data Frequency** | Tick-level (high-frequency) | Hourly (1H) | 🟡 **MEDIUM** - Hourly is fine for your use case |

**Recommendation:**
- ✅ **Keep OHLCV** (LOB data is expensive/complex to obtain)
- ✅ **Add more technical indicators** (you're already doing this)
- ✅ **Consider multi-timeframe features** (combine 1H + 4H + daily)

---

### 2. **Sequence Length** 🟡 MODERATE IMPACT

| Paper | Sequence Length | Your Setup | Recommendation |
|-------|----------------|-----------|----------------|
| Imperial College | **100** (for movement prediction) | **60** | Try **100** for better long-term patterns |
| AI 2021 | Not specified | 60 | Current is fine |
| Fractal 2023 | Not specified | 60 | Current is fine |

**Current:** `sequence_length = 60`
**Recommended:** Try `sequence_length = 100` for classification tasks

---

### 3. **Model Architecture - DLSTM** 🔴 NEEDS IMPROVEMENT

| Component | Imperial College | Your Implementation | Issue |
|-----------|-----------------|-------------------|-------|
| **Trend Extraction** | **AvgPool** (Average Pooling) | **Conv1D** with fixed weights | ⚠️ Different method |
| **Trend LSTM Units** | Same as main (100) | 100 → 50 (reduced) | ⚠️ You're reducing capacity |
| **Remainder LSTM Units** | Same as main (100) | 100 → 50 (reduced) | ⚠️ You're reducing capacity |
| **Fusion** | Simple Add | Add + Dense layer | ✅ Your approach is actually better |
| **Moving Average Window** | Not specified | 10 | ✅ Reasonable |

**Key Issue:** Imperial College uses **AvgPool** for trend extraction, you're using Conv1D.

**Your Current DLSTM:**
```python
# Trend: Conv1D (approximates MA)
trend = MovingAverageLayer(window_size=10)(inputs)  # Uses Conv1D
trend_lstm = LSTM(100 → 50)  # REDUCED CAPACITY
remainder_lstm = LSTM(100 → 50)  # REDUCED CAPACITY
```

**Imperial College DLSTM:**
```python
# Trend: AvgPool
X_t = AvgPool(Padding(X_T))  # Average pooling operation
# Remainder
X_r = X_T - X_t
# Both use FULL units (100)
H_t = LSTM(100)(X_t)
H_r = LSTM(100)(X_r)
# Simple addition
output = Add([H_t, H_r])
```

**Recommendation:**
1. ✅ **Use AvgPool1D** instead of Conv1D for trend extraction
2. ✅ **Keep full units (100)** for both trend and remainder branches
3. ✅ **Remove extra Dense layer** - keep it simple like the paper

---

### 4. **Training Hyperparameters** 🟡 MODERATE IMPACT

| Parameter | Imperial College | AI 2021 | Fractal 2023 | Your Setup | Recommendation |
|-----------|-----------------|---------|--------------|-----------|----------------|
| **Batch Size** | **64** | 32 | Not specified | **32** | Try **64** for classification |
| **Learning Rate** | Not specified | Not specified | Not specified | 0.0001 | ✅ Good |
| **Dropout** | Not specified | 0.2 | **0.1** | 0.2 | ✅ Good |
| **Units** | 100 | 100 | 100 | 100 | ✅ Matches |
| **Layers** | 2 | 2 | 2 | 2 | ✅ Matches |
| **Optimizer** | Adam | Adam | Adam | Adam | ✅ Matches |

**Recommendation:**
- Try `batch_size = 64` for classification (matches Imperial College)

---

### 5. **Classification Labeling** 🟡 MODERATE IMPACT

| Parameter | Imperial College | Your Setup | Status |
|-----------|-----------------|-----------|--------|
| **Smoothing k** | **20, 30, 50, 100** (tested all) | **20** | ⚠️ Only testing k=20 |
| **Delta Threshold** | Fixed (not specified) | **Dynamic (auto)** | ✅ Your approach is better! |
| **Label Method** | Past k vs Future k average | Past k vs Future k average | ✅ Matches |

**Imperial College Results by k:**
- k=20: **73.10% accuracy** (DLSTM)
- k=30: **70.61% accuracy**
- k=50: **67.45% accuracy**
- k=100: **63.73% accuracy**

**Your Results:**
- k=20: **55.8% accuracy** (GRU/DLSTM)

**Recommendation:**
- ✅ **Test k=30, 50, 100** to find optimal smoothing window
- ✅ **Keep dynamic delta** (your innovation is good)

---

### 6. **Technical Indicators** ✅ YOU'RE AHEAD

| Feature | Research Papers | Your Setup | Status |
|---------|----------------|-----------|--------|
| **RSI** | Not used | ✅ Yes | ✅ Good addition |
| **MACD** | Not used | ✅ Yes | ✅ Good addition |
| **Bollinger Bands** | Not used | ✅ Yes | ✅ Good addition |
| **Volume Ratio** | Not used | ✅ Yes | ✅ Good addition |

**Note:** Research papers use **raw OHLCV** or **LOB data**. Your technical indicators are a **good enhancement** that papers don't explore.

---

### 7. **Transaction Costs** 🔴 MISSING

| Aspect | Imperial College | Your Setup | Impact |
|--------|-----------------|-----------|--------|
| **Backtesting** | ✅ Yes (with 0.002% cost) | ❌ **No** | 🔴 **CRITICAL** - Can't verify profitability |
| **Trading Simulation** | ✅ Yes | ❌ **No** | 🔴 **CRITICAL** |
| **Sharpe Ratio** | ✅ Calculated | ❌ **No** | 🔴 **CRITICAL** |

**Imperial College Findings:**
- Without transaction cost: DLSTM achieves 14.97% CPR (Cumulative Price Return)
- **With 0.002% transaction cost**: DLSTM achieves **3.04% CPR** (still profitable!)
- **Bitvavo fees: 0.25%** (125x higher than paper's 0.002%)

**Recommendation:**
- 🔴 **URGENT:** Create backtesting script with **real transaction costs (0.25%)**
- This will show if your model is **actually profitable**

---

### 8. **Data Size** ✅ YOU'RE GOOD

| Aspect | Imperial College | Your Setup | Status |
|--------|-----------------|-----------|--------|
| **Training Data** | 6 days (high-frequency ticks) | **~17,500 hours** (~2 years) | ✅ **Much larger!** |
| **Test Data** | 3 days | ~3,500 hours | ✅ **Good** |

**Your dataset size is excellent!** More data = better generalization.

---

## 🎯 Priority Improvements

### **Priority 1: CRITICAL** 🔴
1. **Fix DLSTM Architecture**
   - Use `AvgPool1D` instead of `Conv1D` for trend extraction
   - Keep full units (100) for both branches (don't reduce to 50)
   - Simplify fusion (remove extra Dense layer)

2. **Create Backtesting Script**
   - Simulate trading with **0.25% transaction costs** (Bitvavo fees)
   - Calculate Sharpe Ratio, Max Drawdown, Total Return
   - This will **prove profitability** (or lack thereof)

### **Priority 2: HIGH** 🟡
3. **Increase Sequence Length**
   - Test `sequence_length = 100` for classification
   - Imperial College used 100 for movement prediction

4. **Test Multiple Smoothing k Values**
   - Test k = 20, 30, 50, 100
   - Find optimal smoothing window for your data

5. **Increase Batch Size**
   - Try `batch_size = 64` for classification tasks

### **Priority 3: MEDIUM** 🟢
6. **Hyperparameter Tuning**
   - Learning rate: Try 0.0005, 0.001 for classification
   - Dropout: Try 0.1 (Fractal paper used this)
   - Units: Try 128, 64 (see what works)

7. **Multi-Timeframe Features**
   - Combine 1H + 4H + daily features
   - Can improve pattern recognition

---

## 📈 Expected Improvements

If you implement Priority 1 fixes:

| Metric | Current | Expected (After Fixes) | Research Paper |
|--------|---------|----------------------|----------------|
| **Accuracy** | 55.8% | **60-65%** | 63-73% |
| **Profitability** | ❓ Unknown | ✅ **Measurable** | Profitable at 0.002% cost |

**Why you're below research:**
1. ❌ Using OHLCV instead of LOB (less information)
2. ❌ DLSTM architecture differs (AvgPool vs Conv1D, reduced units)
3. ❌ Sequence length 60 vs 100
4. ❌ Only testing k=20 (not optimal k)

**But you have advantages:**
1. ✅ Much larger dataset (17,500 vs 6 days)
2. ✅ Technical indicators (papers don't use)
3. ✅ Dynamic delta threshold (better than fixed)

---

## 🚀 Next Steps

1. **Fix DLSTM** (Priority 1) - Should take 30 minutes
2. **Create Backtesting** (Priority 1) - Should take 1-2 hours
3. **Test sequence_length=100** (Priority 2) - Should take 1 hour
4. **Test multiple k values** (Priority 2) - Should take 2-3 hours

**Estimated time to implement all Priority 1 & 2:** 4-6 hours

**Expected result:** Accuracy improvement from 55.8% → 60-65%, plus ability to verify profitability.


