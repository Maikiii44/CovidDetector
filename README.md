# Covid Detector

This project is related to a Kaggle dataset aimed at detecting Covid cases from CT scan images: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

## ⚙️ Installation & Setup

To install it you just need to run the following command in an environment with Python
3.11 or higher:

> `poetry install --all-extras`

In case you want to update the dependencies once installed the first version, you just
need to:

> `poetry update`

And the `poetry.lock` file will be updated too, if applicable.

## 📁 Project Structure

```
📂 covid
├── 📂 Covid19-dataset         - data files
├── 📂 covid                     - the main package
|   ├── 📂 nn                  - code for neural network
|   ├── 🐍 model.py            - neural network definition
|   ├── 🐍 data.py             - data module
|   ├── 🐍 metrics.py          - torchmtrics
|   ├── 🐍 module.py           - ligthtining module
|   └── 🐍 train.py            - trainer to manage the model training
└── 🐍 covid_notebook.ipynb    - the nn in pytorch lightning
```

## 📊 Results

To run the notebook to train the models, it is first necessary to download the dataset and save it in a folder named Covid19-dataset. Then, simply run the notebook.

Additionally, to observe the results in the metrics, it is necessary to start MLflow locally with the command in the terminal at the root of the folder: mlflow ui --backend-store-uri ./mlflow and open the page http://127.0.0.1:5000.