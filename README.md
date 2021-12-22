### Re-implementation for Unsupervised Data Augmentation (UDA)

- Uses EfficientNet-b0 as backbone network
- Uses STL-10 dataset

### Installing environment

- (Recommand) With Docker
Use official Pytorch docker image: `pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel`  
Install EfficientNet: `pip install efficientnet_pytorch`  

- Without Docker
Install Pytorch and EfficientNet: `pip install torch==1.8.0 torchvision==0.9.0 efficientnet_pytorch`


### Get trained model weight

- Install gdown: `pip install gdown`  
- Download weight:  
    - Model without TSA: `gdown --id 1zuB9NsAm34PgSsP9MEhSIEkUmPOUlJDG`  
