# Homework for Building scoring model

### Installation process

Clone the repository
```bash
mkdir Coding_AI
cd Coding_AI
git clone https://github.com/TPODAvia/HW2
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

Because of github that allows only 25Mb of file we need to combine the chuck of datasets to one train dataset.

```bash
python Lab2/combine_small_csv.py
```

For torch libraries we need to go to the official website and install torch with Cuda
https://pytorch.org/get-started/locally/

For Jetson Nano please refer this tutorial:

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


PyTorch offers a wide range of loss functions tailored for various machine learning tasks, including regression, classification, and ranking. Here's an overview of the different types of loss functions available in PyTorch and their usage:

### Regression Losses
- **Mean Absolute Error Loss (L1 Loss)**: Measures the average of the absolute differences between the predicted and actual values. It's robust to outliers and suitable for regression tasks [4].
- **Mean Squared Error Loss (MSE Loss)**: Calculates the average of the squared differences between the predicted and actual values. It's sensitive to outliers but penalizes larger errors more heavily [4].
- **Smooth L1 Loss**: Combines the benefits of MSE and MAE losses. It uses a squared term for errors smaller than a threshold and an absolute term for larger errors, making it less sensitive to outliers [1].
- **Huber Loss**: A combination of MAE and MSE, it switches between the two based on a threshold value, making it more balanced for regression tasks with outliers [4].

### Classification Losses
- **Cross-Entropy Loss**: Suitable for multi-class classification problems, it measures the performance of a classification model whose output is a probability value between  0 and  1 [0][1].
- **Binary Cross-Entropy Loss (BCE Loss)**: Specifically for binary classification problems, where the output of the neural network is a sigmoid layer to ensure the output is close to  0 or  1 [1][3].
- **Binary Cross-Entropy Loss with Logits (BCEWithLogitsLoss)**: Combines a sigmoid layer and BCE loss in one class, providing numerical stability and being more numerically stable than plain Sigmoid followed by BCE Loss [1].
- **Negative Log-Likelihood Loss (NLL Loss)**: Used for multi-class classification problems, it calculates the negative log probability of the correct class [0].
- **Hinge Embedding Loss**: Used in semi-supervised learning tasks or learning nonlinear embeddings, it measures whether two inputs are similar or dissimilar [4].

### Ranking Losses
- **Margin Ranking Loss**: Measures the relative distance between a set of inputs in a dataset, useful for ranking problems [1].
- **Triplet Margin Loss**: Specifically designed for learning embeddings with the aim of using these embeddings to measure similarity between pairs of inputs [0].

### Other Losses
- **Kullback-Leibler divergence**: Measures the difference between two probability distributions, often used in unsupervised learning tasks [0].
- **CTC Loss**: Used in tasks involving sequence prediction, such as speech recognition or handwriting recognition [4].

To use any of these loss functions in PyTorch, you typically import them from the `torch.nn` module and then instantiate the loss function. For example, to use the Mean Squared Error Loss:

```python
import torch.nn as nn

mse_loss = nn.MSELoss()
```

And to compute the loss between predictions and targets:

```python
predictions = torch.randn(10, requires_grad=True)
targets = torch.randn(10)
loss = mse_loss(predictions, targets)
```

Choosing the right loss function depends on the specific task at hand, such as regression, classification, or ranking, and the characteristics of the data, like the presence of outliers.
