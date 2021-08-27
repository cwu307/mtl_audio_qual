# A multitask teacher-student framework for perceptual audio quality assessment

This folder contains the python implementation (referred to as `smaq_cli`) for the following publication:
```
Chih-Wei Wu, Phillip A. Williams, William Wolcott, 
"A multitask teacher-student framework for perceptual audio quality assessment",
in Proceedings of European Signal Processing Conference (EUSIPCO), 2021
```

## Installation

### Step 0: Create a virtual environment with Python 3.7 
We highly recommend setting up a clean environment using `virtualenv` in order to avoid dependency conflicts. 

For instance:
```
>> virtualenv -p python3.7 .venv
```
Note that **Python 3.7** is a requirement for this package. If you don't have Python 3.7 and the above command failed, you could try the followings:
```
>> brew install python@3.7
```
and create your virtual environment with (please change the path to your python3.7 accordingly):
```
>> virtualenv -p /usr/local/opt/python@3.7/bin/python3.7 .venv
```

### Step 1: Activate your virtual environment
Please run the following command to activate your virtual environment:
```
>> source .venv/bin/activate
```

### Step 2: Install smaq_cli package locally
Please navigate to `/code/` folder and run the following commands (be careful with the `.`):
```
(.venv) >> pip install --upgrade pip
(.venv) >> pip install -e .
```

## Usage
To confirm the installation of the package, please try the following commands:

### 1) Get help information 
```
(.venv) >> smaq-cli -h
```

### 2) Run on test files
```
(.venv) >> smaq-cli -t ./smaq_cli/data/tar.wav -r ./smaq_cli/data/ref.wav
```
a successful execution of the above command should return:
```
=============================================================
SMAQ score = 3.5690267086029053
raw score = [[0.7432404  0.5091952  0.7000091  0.64226925]]
raw features = [  0.89797038 -13.11266633   0.93788654   0.86328718]
=============================================================
```

### 3) Import smaq_cli in Python
```
from smaq_cli.smaq_predictor import SmaqPredictor

smaq = SmaqPredictor()
[smaq_score, raw_scores, raw_features] = smaq.predict(smaq.RESOURCE_TAR, smaq.RESOURCE_REF)
```

### 4) Import smaq_cli as a feature extractor in Python
```
from smaq_cli.feature_extractor import SmaqFeatureExtractor

smaq_feat = SmaqFeatureExtractor()
feature_vec = smaq_feat(tar_sig, ref_sig, fs)  # first define your signals & sampling rate
```


## Citing
To cite this work, please use the following:

```
@inproceedings{
  wu_2021,
  title={A multitask teacher-student framework for perceptual audio quality assessment},
  author={Wu, Chih-Wei and Williams, Phillip A and Wolcott, William},
  booktitle={European Signal Processing Conference (EUSIPCO)},
  year={2021}
}
```

## Contact
Chih-Wei Wu \
cwu307[at]gmail.com

