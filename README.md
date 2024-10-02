# 🎶 Music Genre Classification Model 🎧

## 🧑‍💻 Project by Team Penguins in [(SHAI For AI | شاي للذكاء الاصطناعي)](https://www.linkedin.com/company/shaiforai/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_recent_activity_content_view%3BquLv6KkwThWijw6vCr8kew%3D%3D) Training

This project focuses on classifying music genres using **Ensemble Learning Stacking** with a variety of machine learning models. We used **Random Forest, SVC, KNN, and Logistic Regression** as base models, and a **Random Forest** model as the meta-learner (blender). The project involves data preprocessing, feature engineering, and model fine-tuning to achieve optimal performance.

---

## 🔍 Project Overview
In this project, we aim to classify music genres based on various features such as popularity, loudness, danceability, and more. We used an **Ensemble Learning** approach (stacking) to combine the predictions of multiple models to improve accuracy.

---

## 🗂️ Dataset Overview
The dataset contains **14,396 rows** and **15 features**, including the target variable `Class`, which represents the music genre. Key features include:
- **Danceability**: How suitable a track is for dancing.
- **Valence**: The musical positivity conveyed by a track.
- **Energy**: Intensity and activity in a track.
- **Loudness**: Overall sound volume.

---

## 🛠️ Data Processing Steps

### 1️⃣ Exploratory Data Analysis (EDA)
- Inspected feature distributions and identified skewness in some numerical features (e.g., loudness).
- Detected class imbalance in categorical features like time signature.
- Explored relationships between features using scatter and box plots.

### 2️⃣ Data Preprocessing
- **Label Encoding**: Categorical variables were encoded to ensure proper ordering of data.
- **Outlier Handling**: Applied **log transformation** and **Interquartile Range (IQR)** method to handle outliers effectively.
- **Scaling**: Used **StandardScaler** to normalize the data.
  
---

## 🏗️ Feature Engineering
- Created interaction features such as:
  1. **Danceability-Energy Interaction**: Indicates tracks that are both highly danceable and energetic.
  2. **Valence-Energy Product**: Indicates tracks that are both positive and energetic.
- Added average popularity per artist as a new feature to improve model performance.
  
### Clustering
- Applied **K-Means Clustering** to transform continuous numerical features into categorical features, capturing non-linear relationships and further improving model performance.

---

## 🎯 Model Selection & Ensemble Learning
We applied **Ensemble Learning** techniques (stacking and voting) to combine the predictions from multiple base models. 

### Base Models Used:
1. **RandomForestClassifier**: High interpretability and robustness.
2. **SVC (Support Vector Classifier)**: Effective for high-dimensional data.
3. **KNeighborsClassifier**: A non-parametric method useful for smaller datasets.
4. **LogisticRegression**: Simplicity and effectiveness with linear data.

### Stacking Classifier:
- **Meta-Learner (Blender)**: RandomForestClassifier.
- This stacking approach captured complex patterns by aggregating decisions from the base models, yielding an **F1 Score (Weighted)** of **0.5097**.

---

## 🎛️ Model Fine-Tuning
To improve the performance of the Stacking Classifier, we applied **RandomizedSearchCV** to fine-tune the hyperparameters of both the base models and the meta-learner.

- **Random Forest Hyperparameters**: Number of estimators, max depth, and max features.
- **SVC & KNN Hyperparameters**: Kernels, neighbors, etc.
- **Evaluation Metric**: Focused on **F1 Score (Weighted)** to balance precision and recall across all music genres.

After tuning, the model achieved an **F1 Score (Weighted)** of **0.5638**.

---

## 🚀 How to Run the Project
1. Clone the repository.
2. Open the `Final-Music-Genre-Classification-Model.ipynb` file in **VS Code** or **Jupyter Notebook**.
3. Run all the cells to preprocess the data, train the model, and evaluate the results.

```bash
# Clone this repository
git clone https://github.com/yourusername/Music-Genre-Classification-Model.git

# Open the Jupyter Notebook or VS Code to run
```
---

## 📊 Results

Final model performance:

  F1 Score (Weighted): **0.5765**.

The ensemble stacking approach effectively captured relationships between the features and improved classification performance.

---
## 🤝 Contributing

Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature-branch`).
3. Make your changes and test them.
4. Submit a pull request.

---

## 👥 Team Penguins

  - الحارث بشار الحاج حسين
  - فهد صلاح شديد
  - الشيماء محمد المذيب
  - أحمد علي فتحي الشيخ أحمد

---
## Copyrights

Made with 🤍 by **Penguins Team** 

---

## 🌐 Follow Me

- [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/alharth-alhaj-hussein-023417241)  
- [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AlharthAlhajHussein)   
- [![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@Alharth.Alhaj.Hussein)
- [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/alharthalhajhussein)
 
