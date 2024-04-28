# customer-segmentation-ml-model.
# Customer Segmentation ML Model

## Introduction
Customer segmentation is crucial for businesses to understand their target audience and tailor their marketing strategies accordingly. This project utilizes unsupervised machine learning techniques, specifically K-Means clustering, to segment customers based on their demographic profile, interests, and affluence level.

## Overview
This project aims to demonstrate the application of K-Means clustering for customer segmentation. It includes data pre-processing, building the clustering model, evaluating model performance, and interpreting customer segments.

## Steps Followed in Executing the Project

### Step 1: Imports and Reading the Data Frame
- Imported necessary libraries and read the dataset from Google Drive.
- Preprocessed the data by dropping unnecessary columns and standardizing numerical variables.

### Step 2: Standardizing Variables
- Standardized all variables to bring them to a similar scale.

### Step 3: One-Hot Encoding
- Encoded categorical variable 'gender' using one-hot encoding.

### Step 4: Building the Clustering Model
- Visualized the model's performance and determined the optimal number of clusters.
- Calculated silhouette score to evaluate the clustering model.

### Step 5: Silhouette Coefficient
- Explained the silhouette coefficient and calculated it for the model.

### Step 6: Building Clustering Model #2
- Conducted feature selection using Principal Component Analysis (PCA).
- ![image](https://github.com/vaish414/customer-segmentation-ml-model./assets/106098796/babe8787-5a7d-401a-a8a4-cf4f3f3f211e)
- Rebuilt the model with selected principal components and determined the optimal number of clusters.
- Visualized clusters for the second model.

### Step 7: Model 1 vs Model 2
- Compared the performance of the two models and selected the second model for further analysis.
- ![image](https://github.com/vaish414/customer-segmentation-ml-model./assets/106098796/361eedfc-45cf-4b7e-80dc-6670954bedcc)
- ![image](https://github.com/vaish414/customer-segmentation-ml-model./assets/106098796/ae30de01-dadf-480c-9918-1bfaa90d3d4b)


### Step 8: Cluster Analysis
- Mapped clusters back to the dataset and interpreted the segments.
- Visualized clusters based on spending score, annual income, and age.
- Analyzed the main attributes of each cluster and provided insights into customer segments.

## Tools and Technology Used

### I. Python
- Python is a popular programming language for data analysis and machine learning.

### II. Python Libraries:
- Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Kneed: Used for data manipulation, analysis, visualization, and machine learning algorithms.
- Google Colaboratory: A cloud-based platform for machine learning and data analysis.

## Usage
To replicate the project's execution and explore the results, follow these steps:
1. Clone the repository.
2. Open the project notebook in Google Colab or Jupyter Notebook.
3. Execute each cell in the notebook sequentially to reproduce the project workflow.
4. Explore the results and analyze customer segments.

## Dependencies
- Python 3.x
- Required Python libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Kneed

## Future Work
- Explore additional clustering algorithms and compare their performance.
- Incorporate more features or external data sources for better customer segmentation.

## Acknowledgements
- I would like to acknowledge the sources that inspired and guided me during the development of this project. While I cannot locate the original Medium article, I want to express my gratitude to the authors and contributors for their valuable insights and ideas.

## Author
- Vaishnavi Dave(https://github.com/vaish414)
