To reproduce the results in our evaluation section, follow these steps:

1. Extract the data bundle file.
This includes the tokenized javadoc pairs and android pairs that we extracted,
and also their vocabulary file for convinience.
```
7z e bundle.7z
```

2. Install the required dependencies
```
pip3 install -r requirements.txt
```

3. Run the training script
```
python3 train.py
```
This will train two models on three datasets, as described in the paper.
The relevant losses and scores will be printed to standard output.

For the two smaller datasets other than `javadoc`, the `n_epoch_end` parameter in `train.py` needs to be increased to about 50 to reach convergence.
