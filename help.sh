# Install environment
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

# Train and Test
Please using the file **help.sh** for training and testing the ESTIMATE.
```
bash help.sh
```
Follow instructions and it is good to go. If the model is trained, the pretrained folder will exist.  As a result, we can disable the training mode and run the testing mode only.

# Change config
Please go to the file *src/model/estimate/config/final.yaml* and adjust any hyperparameters. A new config file can be created in the same folder and the program **help.sh** will automatically detect it.