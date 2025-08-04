## 🧠 Epileptic Seizure Detection with Random Forest

### ⚡ Powered by BEED Dataset | Streamlit App 🧪

This project builds a machine learning pipeline to classify EEG recordings into four epilepsy-related classes using the **Bangalore EEG Epilepsy Dataset (BEED)**. The model is deployed as a **Streamlit web application** to predict seizure types either via **CSV upload** or **manual input**.

---

### 📂 Dataset: BEED (Bangalore EEG Epilepsy Dataset)

* **Source**: https://archive.ics.uci.edu/dataset/1134/beed:+bangalore+eeg+epilepsy+dataset ,Neurological Research Centre, Bangalore, India
* **Sampling Rate**: 256 Hz
* **Electrode System**: 10-20 standard
* **Duration per segment**: 20 seconds
* **Total segments**: 16,000
* **Subjects**: 80 (20 per category, ages 21–55, gender-balanced)
* **Channels**: X1 to X16 (EEG signals from brain regions)
* **Binary label**: `y` (1 = seizure, 0 = no seizure)
* **Multiclass label**: `label`

  * 0: Healthy
  * 1: Generalized Seizure
  * 2: Focal Seizure
  * 3: Seizure Event (e.g., blinking, nail biting, etc.)

---

### 🧠 Problem Statement

Build a classifier that can distinguish EEG signals into 4 classes:

* Healthy
* Generalized Seizure
* Focal Seizure
* Seizure Events

---

### ✅ Features

* 16 EEG channel inputs (X1 to X16)
* Scalable prediction pipeline using `StandardScaler`
* Trained using `RandomForestClassifier`
* Streamlit UI with:

  * 📄 CSV Upload
  * ⌨️ Manual Feature Entry
* Results displayed with:

  * Class label
  * Expanded explanation
  * Downloadable results

---

### 🛠️ Project Structure

```bash
Epilepsy-Detection/
├── app.py                         # Streamlit App
├── random_forest_beed_model.joblib  # Trained Random Forest model
├── beed_scaler.joblib             # Scaler used during training
├── requirements.txt               # Python dependencies
├── eeg_stage3_focal_seizure.csv   # Sample CSV (Focal Seizure)
├── sample_healthy_eeg.csv         # Sample CSV (Healthy)
├── README.md                      # This file
```

---

### 🔬 Model Performance (Random Forest)

| Model          | Accuracy  |
| -------------- | --------- |
| Random Forest  | **96.0%** |
| Neural Network | 92.1%     |
| SVM            | 74.5%     |

**Classification Report (Random Forest)**:

* Precision: \~0.96
* Recall: \~0.96
* F1 Score: \~0.96

---

### 🚀 How to Run the Project

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/Epilepsy-Detection.git
cd Epilepsy-Detection
```

#### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

#### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

### 🧪 Using the App

#### 📄 Upload CSV

* Prepare a CSV with the following columns: `X1` to `X16`
* Upload it to get predictions for each sample
* Sample files:

  * `sample_healthy_eeg.csv`
  * `eeg_stage3_focal_seizure.csv`

#### ⌨️ Manual Entry

* Fill in the EEG values manually for `X1` to `X16`
* Get an instant prediction with class interpretation

---

### 🧾 Requirements

```
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
```

(Include in `requirements.txt`)

---

### 📸 App Screenshots

| App Home               |  CSV upload              |  Manual Entry              |  Prediction Output and Download  |
| -----------------------| -------------------------| ---------------------------| ---------------------------------|
|   appSS.png            |  appSS2.png              |  appSS1.png                |   appSS1.png                     |
---

### 📌 Notes

* The model is trained on a **balanced dataset**.
* Random Forest was selected for its superior accuracy and robustness.
* All EEG values are **standardized** using `StandardScaler` before model prediction.

---

### 🔒 License

This project is licensed under the MIT License.

---

### 🙋‍♂ Author
Mohammed Abu Hurer
AI Engineering Student | Passionate about Machine Learning, Computer Vision, and Real-World Applications 🚀
Feel free to reach out or contribute!

---

### 🌟 Star this repository
If you found this helpful, give it a ⭐ on GitHub!

---

### 🙏 Acknowledgements

* The creators of the **Bangalore EEG Epilepsy Dataset (BEED)**
* UC Irvine Machine Learning Repository
* [Scikit-learn](https://scikit-learn.org/)
* [Streamlit](https://streamlit.io/)
* OpenAI (for guidance and development assistance)

---
