
# fMRI Depression Classification using Artificial Intelligence

This project focuses on utilizing artificial intelligence (AI) techniques, particularly machine learning (ML) and deep learning (DL), to classify depression in patients based on their functional magnetic resonance imaging (fMRI) data.

## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Modeling Approach](#modeling-approach)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project aims to develop an AI-based model that can classify individuals as either depressed or not depressed based on their brain activity patterns captured through fMRI. The goal is to contribute to the field of mental health diagnostics by providing an automated and non-invasive method for depression detection.

### Key Objectives:
- Preprocess fMRI data for use in machine learning models.
- Build and train classification models to detect depression.
- Evaluate model performance using various metrics like accuracy, precision, recall, and F1-score.

## Data
The dataset used in this project consists of fMRI scans from subjects with and without depression. The data includes brain activity patterns captured during a series of tasks designed to activate regions related to mood regulation, cognitive processing, and emotional response.

### Data Source:
- [Link to dataset](insert link if available)
- Format: `.nii`, `.csv`, `.mat` (depending on your data format)

### Data Preprocessing:
- Motion correction
- Spatial normalization
- Temporal filtering
- Feature extraction (e.g., brain region activation maps)

## Modeling Approach
In this project, various machine learning and deep learning algorithms were explored for classification, including:
- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Random Forest**
- **Convolutional Neural Networks (CNNs)**
- **Recurrent Neural Networks (RNNs)**

Each model was trained on preprocessed fMRI data and evaluated to determine the best approach for classification.

## Setup and Installation

### Prerequisites:
- Python 3.x
- TensorFlow / Keras (for deep learning models)
- Scikit-learn (for machine learning models)
- NiBabel (for handling fMRI data in NIfTI format)
- Matplotlib / Seaborn (for visualization)
- NumPy / Pandas (for data manipulation)

### Installation:
Clone this repository and install the necessary dependencies.

```bash
git clone https://github.com/yourusername/fmri-depression-classification.git
cd fmri-depression-classification
pip install -r requirements.txt
```

## Usage
After setting up the environment, you can run the project with the following command:

```bash
python main.py
```

This will:
1. Load and preprocess the fMRI data.
2. Train the selected machine learning or deep learning model.
3. Output the classification results.

You can also modify the `config.py` file to change model parameters and experiment with different settings.

## Results
The project provides detailed results, including:
- Model accuracy and evaluation metrics (precision, recall, F1-score)
- Confusion matrices and ROC curves for visual evaluation
- Performance comparison of different models

## Contributing
We welcome contributions! If you would like to contribute to this project, feel free to fork the repository, submit issues, or create pull requests.

### How to Contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.
