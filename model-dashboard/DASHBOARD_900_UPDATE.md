# Dashboard Update for 900-Sample Evaluation

## ✅ What Was Updated

### **1. Model Configuration**
Updated to show only 3 models with 900-sample evaluation:
- **Trained ConvNet** (F1: 42.3%)
- **Gemini 2.0 Flash** (F1: 28.6%)
- **Qwen 30B** (F1: 1.6% - needs investigation)

### **2. Data File Paths**
Changed from 250-sample paths to 900-sample paths:
```javascript
// OLD (250 samples)
'/data/eval_results_openai_gpt-5-mini_...'

// NEW (900 samples)
'/data/900/eval_results_trained_convnet.json'
'/data/900/eval_results_gemini_2_0_flash.json'
'/data/900/eval_results_qwen_30b.json'
```

### **3. New Clinical Visualizations**
Added 3 new visualization cards:

#### **🏥 Performance by Comorbidity Burden**
- Compares F1 scores across low, medium, high comorbidity patients
- Shows model performance on complex cases

#### **⚖️ Health Equity Analysis**
- Compares F1 scores across racial/ethnic groups:
  - White
  - Black
  - Asian
  - Hispanic
  - Other
- Critical for fairness assessment

#### **🎚️ Model Calibration**
- Shows Expected Calibration Error (ECE)
- Measures confidence vs accuracy alignment
- Lower ECE = better calibrated

### **4. Dashboard Header**
Updated to reflect 900-sample dataset:
```
ICD-10 Model Comparison Dashboard [900 Samples]
Comprehensive evaluation of Trained ConvNet vs LLMs on ICD-10 coding task.
Includes clinical subgroup analyses: comorbidity, health equity, and calibration metrics.
```

---

## 📊 Available Visualizations (15 Total)

### **Overall Metrics (6)**
1. Summary Table
2. F1 Micro Comparison
3. F1 Macro Comparison
4. Precision Micro Comparison
5. Recall Micro Comparison
6. AUC Micro Comparison

### **Ranking Metrics (3)**
7. Precision@K (K=5,8,15)
8. Recall@K (K=5,8,15)
9. F1@K (K=5,8,15)

### **Clinical Stratifications (5)**
10. Performance by Code Frequency
11. Performance by Document Length
12. **NEW:** Performance by Comorbidity Burden
13. **NEW:** Health Equity Analysis (by race)
14. **NEW:** Model Calibration (ECE)

### **Advanced Analysis (1)**
15. Cross Tab F1 Micro Heatmaps

---

## 🚀 How to Run

### **1. Start Dashboard**
```bash
cd model-dashboard
npm start
```

### **2. Open Browser**
Navigate to: `http://localhost:3000`

---

## 📂 File Structure

```
model-dashboard/
├── public/
│   └── data/
│       ├── 250/          # Old 250-sample data (8 models)
│       │   ├── eval_results_openai_gpt-5-mini_...json
│       │   ├── eval_results_claude_haiku_...json
│       │   └── ...
│       └── 900/          # NEW: 900-sample data (3 models) ✅
│           ├── eval_results_trained_convnet.json
│           ├── eval_results_gemini_2_0_flash.json
│           └── eval_results_qwen_30b.json
├── src/
│   └── components/
│       ├── Dashboard.js   # Updated: 3 new viz cards
│       ├── Dashboard.css  # Updated: dataset badge styling
│       └── ChartView.js   # Updated: handlers for new views
```

---

## 📈 Key Metrics Comparison (900 Samples)

| Model | F1 Micro | P@5 | P@8 | AUC |
|-------|----------|-----|-----|-----|
| **Trained ConvNet** | **42.3%** | **58.5%** | **51.1%** | **72.8%** |
| **Gemini 2.0 Flash** | 28.6% | 40.2% | 35.7% | 64.3% |
| **Qwen 30B** | 1.6% | 2.1% | 1.8% | 51.2% |

---

## 🎯 Clinical Insights (Now Visible in Dashboard!)

### **Comorbidity Burden**
- **ConvNet**: Better on high comorbidity (P@8: 68.1%)
- Shows model handles complex cases well

### **Health Equity**
- Performance across racial groups:
  - White: 53.4%
  - Black: 49.9%
  - Asian: 43.3%
  - Hispanic: 41.9%
  - Other: 47.0%

### **Calibration**
- ConvNet ECE: 17.6% (reasonable)
- Gemini ECE: [see dashboard]
- Lower ECE = better calibration

---

## 🔧 Future Additions

If you evaluate more models on 900 samples:
1. Add results to `/data/900/`
2. Update `modelNames` array in `ChartView.js`
3. Update `fileMap` with new file path
4. Add logo if needed in `logoMap`

---

## 📝 Notes

- **250-sample data** still available in `/data/250/` folder
- To switch back to 250 samples, just revert the changes to `ChartView.js`
- All visualizations now support new clinical subgroups
- Summary table highlights best values per column

---

## ✅ Testing Checklist

- [x] Dashboard loads without errors
- [x] All 15 visualizations accessible
- [x] 3 models showing correct data
- [x] New clinical views display properly
- [x] Summary table includes all metrics
- [x] 900-sample badge visible in header
