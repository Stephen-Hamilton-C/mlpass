# cs4824-project

## Setup

Initialize Git Submodules to load datasets:
```commandline
git submodule init
```

Use Python 3.11 or later.

Setup virtual environment:
- Windows: `py -m venv venv`
- macOS/Linux: `python3 -m venv venv`

Enter virtual environment:
- Windows: `.\venv\Scripts\activate.bat`
- macOS/Linux: `source ./venv/bin/activate`

Install required packages:
```commandline
pip install -r requirements.txt
```

## Run

```commandline
python3 mlpass/main.py
```

## Datasets

The `kaggle` directory contains the password strength dataset retrieved from
[this dataset submitted to Kaggle](https://www.kaggle.com/datasets/bhavikbb/password-strength-classifier-dataset).
It has 3 classifications: 0 (weak), 1 (medium), and 2 (strong)

The `pwlds` directory is a git submodule containing the PWLDS password strength dataset.
This has 5 classifications: 0 (very_weak), 1 (weak), 2 (average), 3 (strong), 4 (very_strong)

The classification set from `pwlds` is used. The `kaggle` dataset is mapped to this classification using the following:
|    `kaggle`   |     `pwlds`     |
| ------------- | --------------- |
|   0 (weak)    |  0 (very_weak)  |
|  1 (medium)   |  2 (average)    |
|  2 (strong)   | 4 (very_strong) |


## Credits
```
Dataset Title: Password Weakness and Level Dataset (PWLDS)
Author: Infinitode Pty Ltd
Date: 2024
Source: https://github.com/Infinitode/PWLDS
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
```
