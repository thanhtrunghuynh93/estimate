# Efficient Integration of Multi-Order Dynamics and Internal Dynamics in Stock Movement Prediction

[![arXiv](https://img.shields.io/badge/arXiv-2211.07400-b31b1b.svg)](https://arxiv.org/abs/2211.07400)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Advances in deep neural network (DNN) architectures have enabled new prediction techniques for stock market data. Unlike other multivariate time-series data, stock markets show two unique characteristics: (i) multi-order dynamics, as stock prices are affected by strong non-pairwise correlations (e.g., within the same industry); and (ii) internal dynamics, as each individual stock shows some particular behaviour. Recent DNN-based methods capture multi-order dynamics using hypergraphs, but rely on the Fourier basis in the convolution, which is both inefficient and ineffective. In addition, they largely ignore internal dynamics by adopting the same model for each stock, which implies a severe information loss. In this paper, we propose a framework for stock movement prediction to overcome the above issues. Specifically, the framework includes temporal generative filters that implement a memory-based mechanism onto an LSTM network in an attempt to learn individual patterns per stock. Moreover, we employ hypergraph attentions to capture the non-pairwise correlations. Here, using the wavelet basis instead of the Fourier basis, enables us to simplify the message passing and focus on the localized convolution. Experiments with US market data over six years show that our framework outperforms state-of-the-art methods in terms of profit and stability.

Please read and cite our paper: [![arXiv](https://img.shields.io/badge/arXiv-2211.07400-b31b1b.svg)](https://arxiv.org/abs/2211.07400)

#### Citation 
```
@inproceedings{huynh2023wsdm,
  author    = {Thanh Trung Huynh and Minh Hieu Nguyen and Thanh Tam Nguyen and Phi Le Nguyen and Matthias Weidlich and Quoc Viet Hung Nguyen and Karl Aberer},
  title     = {Efficient Integration of Multi-Order Dynamics and Internal Dynamics in Stock Movement Prediction},
  booktitle = {{WSDM} '23: The Sixteeth {ACM} International Conference on Web Search
               and Data Mining, Virtual Event / Singapore, February 27 - March 3,
               2023},
  publisher = {{ACM}},
  year      = {2023},
}
```

## Install environment
Init environment using conda
```
conda create -n estimate python=3.8.13
conda activate estimate
```
Install pytorch
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Install torch geometric: Please follow [these instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
For our environment, we use the command:
```
conda install pyg -c pyg
```
Install other packages:
```
pip install -r requirements.txt
```

## Train and Test
Please using the file **help.sh** for training and testing the ESTIMATE.
```
bash help.sh
```
Follow instructions and it's good to go. If the model is trained, the pretrained folder will exist.  As a result, we can disable the training mode and run the testing mode only.
