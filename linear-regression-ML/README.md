# Automobile Fuel Efficiency Prediction Using Linear Regression

This repository contains Project 1 for COE 379L - Software Design for Responsible Intelligent Systems - at the University of Texas. The goal of this project was to predict the fuel efficiency of automobiles using a linear regression machine learning model. The repository includes a Jupyter Notebook (`automobiles_linearRegressionModel.ipynb`) detailing the data preparation, model training, and analysis process, along with a  PDF report summarizing the project's process and findings.

## Project Overview

The project focuses on applying exploratory data analysis (EDA) and machine learning techniques to predict car fuel efficiency. It uses various Python libraries such as pandas for data manipulation, seaborn and matplotlib for data visualization, and scikit-learn for model implementation.

### Data Preparation

The analysis began with the automobiles dataset, which includes 398 entries and variables like mpg, cylinders, displacement, and horsepower. The preparation phase involved cleaning the data, handling missing values, dropping irrelevant columns, and one-hot encoding categorical data to make the dataset suitable for linear regression modeling.

### Linear Regression Model Fitting

The project involved training a linear regression model to predict the fuel efficiency (mpg) of a vehicle based on its attributes. The process included splitting the data into training and testing sets, fitting the model to the training data, and evaluating its performance on testing data.

### Model Analysis

The model's effectiveness was assessed using the R² score, showing a fairly strong preformance in both training and testing phases. This suggests that the model has successfully captured the underlying patterns without overfitting, making it a reliable tool for predicting fuel efficiency.


The project produced a linear regression model that can accurately estimate the fuel efficiency of automobiles.

## Resources

- Data Source: [Automobile Dataset](https://raw.githubusercontent.com/joestubbs/coe379L-sp24/master/datasets/unit01/project1.data)
- ChatGPT: Used for visualization advice (giving me differnt plots to use, formatting plots, etc.). This use is noted within the Jupyter Notebook.

