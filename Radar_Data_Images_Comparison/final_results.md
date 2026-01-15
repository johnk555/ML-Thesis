PS C:\Users\karel\Desktop\ML-Thesis\Code\Radar_Data_Images_Comparison> & C:\Users\karel\AppData\Local\Programs\Python\Python312\python.exe c:/Users/karel/Desktop/ML-Thesis/Code/Radar_Data_Images_Comparison/evaluate_failures.py
--- Loading Models on cuda ---
Loading weights...
--- Scanning Dataset: C:/Users/karel/Automotive ---
Scanning dataset for paired Image+Radar files...
Fusion Dataset Ready. Found 16940 valid pairs.
Data Loaded. Evaluating 16940 samples...
--- Starting Evaluation ---
Processed 16672/16940 samples...

============================================================
📊 EVALUATION SUMMARY
============================================================
Image Model Accuracy: 33.68% (Avg Confidence: 97.39%)
Radar Model Accuracy: 86.64% (Avg Confidence: 96.97%)
Total Samples: 16940
Failed Predictions: 12185
Suspicious (Low Confidence): 31

============================================================
CONFUSION MATRIX - IMAGE MODEL
============================================================
               Person    Cyclist        Car
    Person          0      10883          0
   Cyclist          0       3020          0
       Car          0        351       2686

============================================================
CONFUSION MATRIX - RADAR MODEL
============================================================
               Person    Cyclist        Car
    Person       9577        603        703
   Cyclist          1       2070        949
       Car          4          4       3029

============================================================
DETAILED CLASSIFICATION REPORT - IMAGE MODEL
============================================================
C:\Users\karel\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\karel\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
C:\Users\karel\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
              precision    recall  f1-score   support

      Person     0.0000    0.0000    0.0000     10883
     Cyclist     0.2119    1.0000    0.3497      3020
         Car     1.0000    0.8844    0.9387      3037

    accuracy                         0.3368     16940
   macro avg     0.4040    0.6281    0.4294     16940
weighted avg     0.2171    0.3368    0.2306     16940


============================================================
DETAILED CLASSIFICATION REPORT - RADAR MODEL
============================================================
              precision    recall  f1-score   support

      Person     0.9995    0.8800    0.9359     10883
     Cyclist     0.7733    0.6854    0.7267      3020
         Car     0.6471    0.9974    0.7849      3037

    accuracy                         0.8664     16940
   macro avg     0.8066    0.8543    0.8159     16940
weighted avg     0.8960    0.8664    0.8716     16940


✅ Reports saved:
   - failures_report.txt (12185 samples)
   - suspicious_report.txt (31 samples)

💡 Review 'suspicious_report.txt' for potential labeling errors!