# Efficient Integration of Multi-Order Dynamics and Internal Dynamics in Stock Movement Prediction

This codebase contains the python scripts for ESTIMATE, the model for the submitting WSDM 2023 paper. The full code will be released after the full peer review.

## Environment & Installation Steps
Python 3.6, Pytorch, Pytorch-Geometric and networkx.


```python
pip install -r requirements.txt
```

## Dataset and Preprocessing 

Download the dataset and follow preprocessing steps from [here]. 


## Run

Execute the following python command to train ESTIMATE: 
```python
python train_nyse.py -m NYSE -l 16 -u 64 -a 1 -e NYSE_rank_lstm_seq-8_unit-32_0.csv.npy 
python train_tse.py
python train_nasdaq.py -l 16 -u 64 -a 0.1
```

