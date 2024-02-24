# Homework for Building scoring model

### Installation process

Clone the repository
```bash
mkdir Coding_AI
cd Coding_AI
git clone //sdcfvadf
```
Initiate the virtual environment
```bash
python -m venv venv
```

Install libraries
```bash
cd HW2
pip install -r requirements.txt
```

For torch libraries we need to go to the official website and install torch with Cuda
https://pytorch.org/get-started/locally/

### Usage

1. Modify the `/Lab*/lab*_prep.py` based on your dataset.

2. Run the training:

```bash
python train.py
```

The result you get is `model.pth` and `label_encoder_dict.joblib` for the lab1

3. Test the result

```bash
python test.py
```

4. Submit the result

```bash
python submission.py
```