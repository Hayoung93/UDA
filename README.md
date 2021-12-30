### Re-implementation for Unsupervised Data Augmentation (UDA)

- Uses EfficientNet-b0 as backbone network
- Uses STL-10 dataset

### Installing environment

- (Recommand) With Docker  
Use official Pytorch docker image: `pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel`  
Install EfficientNet, tqdm: `pip install efficientnet_pytorch tqdm`  

- Without Docker  
Install Pytorch, EfficientNet and tqdm: `pip install torch==1.8.0 torchvision==0.9.0 efficientnet_pytorch tqdm`  

### Get trained model weight

- Install gdown: `pip install gdown`  
- Download weight:  
    - Model without TSA 1 (test acc 0.834875): `gdown --id 1zuB9NsAm34PgSsP9MEhSIEkUmPOUlJDG`  
    - Model without TSA 2 (test acc 0.84175): `gdown --id 1YkdY05vXSa5pqojjNzCauCueqn-QdENm`

### Train on STL-10 dataset

- `python train_eff_uda.py`

### TODO:

- Support CIFAR, SVHN dataset
- Add configuration file
