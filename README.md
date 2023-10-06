Titanic Dataset Analysis and Logistic Regression Model
This project involves the analysis of the Titanic dataset and the creation of a logistic regression model to predict survival.

Overview
The Titanic dataset is a well-known dataset in the field of data science and machine learning. It contains information about passengers on board the Titanic, including their age, gender, ticket class, and whether they survived or not. In this project, we will perform data analysis and build a logistic regression model to predict passenger survival based on the available features.

Prerequisites
Before running the code, make sure you have the following dependencies installed:

Python
Pandas
NumPy
Matplotlib
Seaborn
scikit-learn (for Logistic Regression)
You can install these libraries using pip: pip install pandas numpy matplotlib seaborn scikit-learn

Usage
Data Loading

The code begins by loading the Titanic dataset from a CSV file named "titanic_ds.csv."

Data Exploration

The first 5 rows of the dataset are displayed to give an initial overview.
The number of rows and columns in the dataset is printed.
Missing values in each column are checked and displayed.
Unnecessary columns (Cabin and Ticket) are dropped from the dataset.
One-hot encoding is applied to the "Sex" and "Embarked" columns to convert categorical variables into numerical ones.
Missing values in the "Age" column are filled with the median age.
The code then displays the dataset's summary statistics, including measures like mean, standard deviation, minimum, and maximum values for numerical columns.
The counts of survival (0 = No, 1 = Yes) are displayed to understand the distribution of survival in the dataset.
Data Visualization

Several plots are created using Seaborn to visualize the data, including a count plot of survival, gender distribution, survival count by gender, and passenger class distribution.
Data Splitting

The dataset is split into training and testing sets using the train_test_split function from scikit-learn.
Model Building

A Logistic Regression model is created and trained using the training data.
Model Evaluation

The model's accuracy is evaluated on both the training and test datasets, and the results are displayed.
