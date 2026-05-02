# Persian MNIST CNN Homework

This project is organized as a normal Python package and can be run locally, on Kaggle, or on Colab.

## Project structure

```text
homework_code/
├── data/
│   ├── data_loader.py
│   └── dataset/mnist.pkl.gz
├── models/
│   ├── blocks.py
│   └── model.py
├── scripts/
│   ├── main.py
│   ├── train.py
│   └── evaluate.py
├── utils/
│   ├── metrics.py
│   └── visualization.py
├── outputs/
├── saved_models/
└── requirements.txt
```

## Local smoke test

From the project root:
python -m scripts.main --epochs 1 --batch-size 128 --num-workers 0 --block A