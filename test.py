import os
import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="stl-10")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--model_name", default="efficientnet-b0")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--datadir", type=str, default="/data/data/stl10")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--weight", type=str, default="/data/weights/hayoung/eff_uda_noTSA/t2/model_last.pth")
    args = parser.parse_args()

    return args


def main(args):
    stl10_tst_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = datasets.STL10(args.datadir, "test", transform=stl10_tst_transform, download=True)
    testloader = DataLoader(testset, args.batch_size, False, num_workers=0, pin_memory=True)

    supcriterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet.from_name(args.model_name)
    model._fc = nn.Linear(model._fc.in_features, args.num_classes)
    model.to(device)
    model = load_checkpoint(model, args.weight)

    test_loss, test_acc = test(model, testloader, supcriterion)
    print("Test loss: {}\tTest acc: {}".format(test_loss, test_acc))


def test(model, loader, criterion):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(loader)):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            running_acc = (outputs.argmax(1) == labels).sum().item()
            test_acc += running_acc
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_loss /= len(loader)
    test_acc /= len(loader.dataset)
    return test_loss, test_acc


def load_checkpoint(model, weight_path):
    cp = torch.load(weight_path)
    model.load_state_dict(cp["state_dict"])
    return model


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    main(args)
