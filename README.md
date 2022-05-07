# CS444-Proj
Project for CS 444 - Model Selection for Prostate Cancer Imaging Deep Networks

# Data
You can download the data for the PANDA dataset using the kaggle API with the command `kaggle competitions download -c prostate-cancer-grade-assessment` (Warning: This is a pretty large dataset with about 400GB worth of high resolution images). You can preprocess the data into more manageable 128x128 grids by using this handy Kaggle Notebook:  https://www.kaggle.com/code/iafoss/panda-16x128x128-tiles/notebook. After putting your processed data into a folder like 'data/train/train_data_cleaned', you can adjust the code in the Jupyter notebook to your desired optimizer (eg: SGD or Adam) and loss function (eg: Cross Entropy, MSE, or QWK).

