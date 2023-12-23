# Predictions of mnist dataset using CNN

Classification problem of digit recognition from [MNIST dataset](https://pytorch.org/vision/0.15/generated/torchvision.datasets.MNIST.html)

### Preparing for usage

#### 1. Poetry

This tool needs to download all `python` dependencies for this project

To download all dependencies create/activate `python` environment and print:
```shell
poetry install
```
All dependencies will be downloaded

### Training
To start training model move to `digit-recognition`:
```shell
cd digit-recognition
```
and run
```shell
python train.py
```

#### Logging

All training process will be logged on `Wandb` and `MLflow`

### Inferring

To start training model move to `digit-recognition`:
```shell
cd digit-recognition
```
and run
```shell
python train.py
```

`python infer.py` - to get predictions

At the end of inferring all results will be logged on `Wandb` and `MLflow`,
model converted to `.onnx`
