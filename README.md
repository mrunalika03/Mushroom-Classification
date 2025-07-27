# 🍄 Mushroom Classification using Machine Learning

This project uses machine learning to classify mushrooms as **edible** or **poisonous** based on their physical characteristics.

---

## 📂 Dataset

- **Source**: UCI Machine Learning Repository
- **Size**: 8124 records, 23 categorical features
- **Target variable**: `class` (`e` = edible, `p` = poisonous)

---

## 🛠️ Tools & Libraries Used

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 📊 Machine Learning Model

- **Model Used**: Random Forest Classifier
- **Preprocessing**: Label Encoding for categorical variables
- **Accuracy**: ✅ 100% on test data
- **Important Features**:
  - `spore-print-color`
  - `odor`
  - `gill-size`
  - `stalk-shape`

---

## 📈 Visualizations

- Confusion Matrix  
- Feature Importance Bar Graph

---

## 🧪 Project Workflow

1. **Import Libraries**
2. **Load and Preview Dataset**
3. **Data Cleaning & Encoding**
4. **Train-Test Split (70/30)**
5. **Model Training**
6. **Evaluation (Accuracy, Confusion Matrix, Report)**
7. **Feature Importance Visualization**

---

## ✅ Output Snapshot

```bash
Accuracy: 1.0

              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1257
           1       1.00      1.00      1.00      1181
