# Model Calibration & Reliability Analysis

Most machine learning classifiers (like **XGBoost** or **Neural Networks**) are optimized to minimize a loss function (like Log-Loss) or maximize separation (like AUC), but they don't always produce **reliable probabilities**. 

The need for calibration becomes critical in two main scenarios:

1. **Model Architecture Biases:**
    * **XGBoost** tends to push probabilities toward 0 or 1 due to the nature of decision trees, or it may produce "S-shaped" distortions.
    * **Neural Networks** often suffer from **overconfidence**, predicting probabilities near 100% even when the true accuracy is much lower.

2. **Class Imbalance Compensation (Weighting):**
    * When using parameters like `scale_pos_weight` (XGBoost) or `pos_weight` in `BCEWithLogitsLoss` (PyTorch) to handle imbalanced datasets, we artificially inflate the importance of the minority class. 
    * While this improves the model's ability to separate classes (AUC), it **distorts the absolute probabilities**. The model will systematically over-predict the likelihood of the minority class. 
    * **Calibration acts as the "antidote"**, re-aligning these biased outputs with the real-world prevalence without losing the discriminative power gained from weighting.

**Calibration** is the process of adjusting these "raw" scores so they represent real-world frequencies. This is critical when model outputs are used for **risk assessment**, **expected value calculations**, or **clinical/business decision-making** where the exact probability matters, not just the final classification.

## 1. Diagnostic Visualizations

### **Reliability Curve (Calibration Plot)**
* **Perfect Calibration:** The curve follows the $y = x$ diagonal.
* **Sigmoid Shape ("S-shape"):** Common in boosting models (XGBoost). The model is often under-confident near the edges and over-confident in the middle.
* **Below the Diagonal:** The model is **over-confident** (e.g., predicts 0.9 when the true frequency is 0.7). Typical in deep Neural Networks.
* **Above the Diagonal:** The model is **under-confident**.

### **Probability Histograms**
* **Distribution Shift:** Visualizes how the calibration method (Platt, Isotonic, or Temperature) redistributes the "mass" of predictions.
* **Certainty peaks:** High-performing models (High BSS) show sharp peaks near 0 and 1. Calibration ensures these peaks are "honest" and not due to overfitting.

### **Raw vs. Calibrated Mapping**
* **Identity Line ($y=x$):** Indicates the calibrator did not change the original probabilities.
* **Step Functions:** Characteristic of **Isotonic Regression**. It groups ranges of raw probabilities into "bins," often leading to "staircase" effects in Partial Dependence Plots (PDP).
* **Smooth Curves:** Characteristic of **Platt Scaling** and **Temperature Scaling**. These maintain a continuous relationship between raw and calibrated outputs, preserving smoothness in PDPs.

---

## 2. Key Metrics: Brier Score (BS) & Brier Skill Score (BSS)

* **Brier Score (BS):** The Mean Squared Error of probabilities: $BS = \frac{1}{N} \sum (f_i - y_i)^2$. **Lower is better.**
    * *Baseline:* A score of 0.25 is the limit for balanced classes (random 50/50 guess).
* **Brier Skill Score (BSS):** Measures the relative improvement over a "naive" baseline (predicting the mean prevalence).
    * $BSS = 1 - \frac{BS_{model}}{BS_{baseline}}$
    * **$BSS > 0$:** The model provides value over the mean.
    * **$BSS \approx 1$:** Near-perfect probabilistic predictions.

---

## 3. Calibration Methods Comparison

| Method | Type | Best For... | Key Characteristics |
| :--- | :--- | :--- | :--- |
| **Platt Scaling** | Parametric (Logistic) | Small datasets ($N < 1000$ in val) | Rigid, smooth, prevents overfitting but can't fix complex distortions. |
| **Isotonic Regression** | Non-parametric (Isotonic) | Large datasets, Trees (XGBoost) | Flexible, handles any monotonic error. Can introduce "steps" (discontinuities). |
| **Temperature Scaling** | Parametric (Single-param $T$) | Deep Neural Networks | Preserves ranking (AUC), only adjusts confidence/softness of logits. |

---

## 4. Decision Rule

The primary metric for model selection is the **Brier Skill Score (BSS)** evaluated on the **Test Set**.

1. **Optimal Calibration:** Choose the method (Platt, Isotonic, or Temperature) that yields the **highest $BSS_{test}$**.
2. **Threshold for Calibration:** If $BSS_{calibrated} \leq BSS_{raw}$, maintain the **Raw Model**. This indicates that the base model is already well-calibrated or that the calibration process is introducing noise.
3. **Consistency Check:** If $BSS_{val} \gg BSS_{test}$, the calibrator is **overfitting** the validation set (frequent with Isotonic Regression on small samples). In this case, prefer a more rigid method like Platt Scaling or no calibration at all.