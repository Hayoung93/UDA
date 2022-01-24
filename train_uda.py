import argparse
import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import models
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from torch_ema import ExponentialMovingAverage
from randaugment import RandAugment
from utils.misc import SharpenSoftmax, get_tsa_mask, load_full_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="stl10")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--model_arch", default="efficientnet")
    parser.add_argument("--model_name", default="efficientnet-b0")
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--results_dir", type=str, default="/data/weights/hayoung/uda/t1")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--datadir", type=str, default="/data/data/stl10")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--lr", type=float, default=0.015)
    parser.add_argument("--tensorboard_path", type=str, default="./runs/uda/t1")
    parser.add_argument("--tsa", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--weight", type=str, default="/data/weights/hayoung/uda/t1/model_last.pth")
    parser.add_argument("--resolution", type=int, default=224)
    args = parser.parse_args()

    return args


def get_loaders(args, resolution, train_transform=None, test_transform=None):
    # transforms
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Pad(12),
            transforms.RandomCrop(96),
            transforms.Resize(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    train_transform_u = transforms.Compose([
        transforms.Resize(resolution),
        transforms.PILToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # dataset
    if args.dataset_name == "stl10":
        valset = datasets.STL10(args.datadir, "test", transform=test_transform, download=True)
        trainset = datasets.STL10(args.datadir, "train", transform=train_transform, download=True)
        unlabelset = datasets.STL10(args.datadir, "unlabeled", transform=train_transform_u, download=True)
    else:
        raise Exception("Not supported dataset")
    # loader
    trainloader = DataLoader(trainset, args.batch_size, True, num_workers=0, pin_memory=True)
    valloader = DataLoader(valset, args.batch_size, False, num_workers=0, pin_memory=True)
    trainloader_u = DataLoader(unlabelset, args.batch_size, True, num_workers=0, pin_memory=True)

    return trainloader, trainloader_u, valloader


def get_model(args, device):
    if args.model_arch == "efficientnet":
        model = EfficientNet.from_name(args.model_name)
        model._fc = nn.Linear(model._fc.in_features, args.num_classes)
    elif args.model_arch == "wideresnet":
        model = eval("models." + args.model_name + "(pretrained=False)")  # wide_resnet50_2
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    else:
        raise Exception("Not supported network architecture")

    # if args.ema:
    #     ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    #     ema = ema.to(device)
    # else:
    #     ema = None

    model.to(device)
    return model

def main(args):
    writer = SummaryWriter(args.tensorboard_path)
    if not os.path.exists(args.tensorboard_path):
        os.mkdir(args.tensorboard_path)
    
    resolution = (args.resolution, args.resolution)
    train_transform_u = transforms.Compose([
        transforms.Resize(96),
        transforms.Pad(12),
        transforms.RandomCrop(96),
        transforms.Resize(resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        RandAugment(1, 2),
    ])
    trainloader, trainloader_u, valloader = get_loaders(args, resolution)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args, device)
    ema = None  # temporarily None
    
    supcriterion = nn.CrossEntropyLoss(reduction='none')
    unsupcriterion = nn.KLDivLoss(reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    if args.resume:
        model, optimizer, scheduler, last_epoch, best_val_loss, best_val_acc = \
            load_full_checkpoint(model, optimizer, scheduler, args.weight)
        print("Loaded checkpoint from: {}".format(args.weight))
        start_epoch = last_epoch + 1
    else:
        start_epoch = 0
        best_val_acc = 0.
    
    for ep in range(start_epoch, args.num_epochs):
        scheduler.step()
        print("Epoch {} --------------------------------------------".format(ep + 1))
        train_loss, train_acc, model, optimizer = \
            train(ep, model, trainloader, trainloader_u, train_transform_u,
                  supcriterion, unsupcriterion, optimizer, writer, args.num_epochs, ema)
        print("Train loss: {}\tTrain acc: {}".format(train_loss, train_acc))
        val_loss, val_acc = eval_model(ep, model, valloader, supcriterion, writer, args.num_epochs, ema)
        print("Val loss: {}\tVal acc: {}".format(val_loss, val_acc))
        print("--------------------------------------------")
        # scheduler.step(val_loss)
        print("{}".format(optimizer.state_dict))
        if ep == 0:
            best_val_loss = val_loss
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                save_checkpoint(ep, model, optimizer, scheduler, args.results_dir, best_val_loss, best_val_acc, True)
        save_checkpoint(ep, model, optimizer, scheduler, args.results_dir, best_val_loss, best_val_acc, False)
    print("Best Val Loss: {} / Acc: {}".format(best_val_loss, best_val_acc))


def train(ep, model, suploader, unsuploader, unsuptransform, supcriterion, unsupcriterion, optimizer, writer, eps, ema):
    model.train()
    train_loss = 0
    train_acc = 0
    sup_generator = iter(suploader)
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    softmax = nn.Softmax(dim=1)
    sharpen_softmax = SharpenSoftmax(0.4, dim=1)
    for i, (inputs, _) in enumerate(tqdm(unsuploader)):
        optimizer.zero_grad()
        try:
            sup_inputs, labels = next(sup_generator)
        except StopIteration:
            sup_generator = iter(suploader)
            sup_inputs, labels = next(sup_generator)
        sup_inputs, labels = sup_inputs.cuda(), labels.cuda()
        unsup_inputs = normalize(inputs.cuda() / 255.)
        unsup_aug_inputs = normalize(unsuptransform(inputs).cuda() / 255.)

        # forward
        sup_outputs = model(sup_inputs)
        unsup_aug_outputs = model(unsup_aug_inputs)
        with torch.no_grad():
            unsup_outputs = model(unsup_inputs)

        # backward
        sup_loss = supcriterion(sup_outputs, labels)
        if args.tsa:
            tsa_mask = get_tsa_mask(sup_outputs, eps, ep, len(unsuploader), i)
            sup_loss = (sup_loss * tsa_mask.max(1)[0]).sum()
        else:
            sup_loss = sup_loss.mean()
        unsup_pred = sharpen_softmax(unsup_outputs)
        confidence_mask = (unsup_pred.max(dim=-1)[0] > 0.7).float()  # beta 0.7
        unsup_loss = unsupcriterion(unsup_pred, softmax(unsup_aug_outputs))
        unsup_loss = confidence_mask.unsqueeze(1) * unsup_loss
        unsup_loss = unsup_loss.mean()
        full_loss = sup_loss + unsup_loss  # lambda = 1
        writer.add_scalar("train loss", full_loss.item(), ep * len(unsuploader) + i)
        train_loss += full_loss.item()
        full_loss.backward() 
        optimizer.step()
        if ema is not None:
            ema.update()
        
        running_acc =(sup_outputs.argmax(1) == labels).sum().item() 
        train_acc += running_acc
        writer.add_scalar("train acc", running_acc, ep * len(unsuploader) + i)
    train_loss /= len(unsuploader)
    train_acc /= len(unsuploader.dataset)
    return train_loss, train_acc, model, optimizer


def eval_model(ep, model, loader, criterion, writer, eps, ema):
    model.eval()
    val_loss = 0
    val_acc = 0
    if ema is not None:
        with ema.average_parameters():
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(loader):
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    running_acc = (outputs.argmax(1) == labels).sum().item()
                    val_acc += running_acc
                    loss = criterion(outputs, labels).sum()
                    val_loss += loss.item()
                    writer.add_scalar("val loss", loss.item(), ep * len(loader) + i)
                    writer.add_scalar("val acc", running_acc, ep * len(loader) + i)
    else:
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                running_acc = (outputs.argmax(1) == labels).sum().item()
                val_acc += running_acc
                loss = criterion(outputs, labels).sum()
                val_loss += loss.item()
                writer.add_scalar("val loss", loss.item(), ep * len(loader) + i)
                writer.add_scalar("val acc", running_acc, ep * len(loader) + i)
    val_loss /= len(loader)
    val_acc /= len(loader.dataset)
    return val_loss, val_acc


def save_checkpoint(ep, model, optimizer, scheduler, savepath, best_val_loss, best_val_acc, isbest):
    save_dict = {
        "epoch": ep,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc
    }
    if isbest:
        torch.save(save_dict, os.path.join(savepath, "model_best.pth"))
    else:
        torch.save(save_dict, os.path.join(savepath, "model_last.pth"))


if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    main(args)