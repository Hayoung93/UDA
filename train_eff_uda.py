import argparse
import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from randaugment import RandAugment
from utils.misc import SharpenSoftmax, get_tsa_mask


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="stl-10")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--model_name", default="efficientnet-b0")
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--results_dir", type=str, default="/data/weights/hayoung/eff_uda_noTSA/t2")
    parser.add_argument("--randaug", action="store_true")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--datadir", type=str, default="/data/data/stl10")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--tensorboard_path", type=str, default="./runs/eff_uda_noTSA/t2")
    parser.add_argument("--tsa", action="store_true", default=True)
    args = parser.parse_args()

    return args


def main(args):
    writer = SummaryWriter(args.tensorboard_path)
    if not os.path.exists(args.tensorboard_path):
        os.mkdir(args.tensorboard_path)
    stl10_transform = transforms.Compose([
            transforms.Pad(12),
            transforms.RandomCrop(96),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    stl10_unlabel_transform = transforms.Compose([
            transforms.Pad(12),
            transforms.RandomCrop(96),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            RandAugment(1, 2),
            # transforms.ConvertImageDtype(torch.float32),
        ])
    stl10_tst_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if args.randaug == True:
        stl10_transform.transforms.insert(-3, RandAugment(1, 2))
    trainset = datasets.STL10(args.datadir, "train", transform=stl10_transform, download=True)
    valset = datasets.STL10(args.datadir, "test", transform=stl10_tst_transform, download=True)
    trainloader = DataLoader(trainset, args.batch_size, True, num_workers=0, pin_memory=True)
    valloader = DataLoader(valset, args.batch_size, False, num_workers=0, pin_memory=True)
    unlabelset = datasets.STL10(args.datadir, "unlabeled", transform=transforms.PILToTensor(), download=True)
    unlabelloader = DataLoader(unlabelset, args.batch_size, True, num_workers=0, pin_memory=True)
    # trainunlabelset = datasets.STL10(args.datadir, "train+unlabeled", transform=stl10_transform, download=True)
    # trainunlabelloader = DataLoader(trainunlabelset, args.batch_size, True, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet.from_name(args.model_name)
    model._fc = nn.Linear(model._fc.in_features, args.num_classes)
    model.to(device)
    supcriterion = nn.CrossEntropyLoss(reduction='none')
    unsupcriterion = nn.KLDivLoss(reduction="none")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)
    
    for ep in range(args.num_epochs):
        scheduler.step()
        print("Epoch {} --------------------------------------------".format(ep + 1))
        train_loss, train_acc, model, optimizer = \
            train(ep, model, trainloader, unlabelloader, stl10_unlabel_transform,
                  supcriterion, unsupcriterion, optimizer, writer, args.num_epochs)
        print("Train loss: {}\tTrain acc: {}".format(train_loss, train_acc))
        val_loss, val_acc = eval(ep, model, valloader, supcriterion, writer, args.num_epochs)
        print("Val loss: {}\tVal acc: {}".format(val_loss, val_acc))
        print("--------------------------------------------")
        # scheduler.step(val_loss)
        print("{}".format(optimizer.state_dict))
        if ep == 0:
            best_val_loss = val_loss
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(ep, model, optimizer, scheduler, args.results_dir, True)
                best_val_acc = val_acc
        save_checkpoint(ep, model, optimizer, scheduler, args.results_dir, False)
    print("Best Val Loss: {} / Acc: {}".format(best_val_loss, best_val_acc))


def train(ep, model, suploader, unsuploader, unsuptransform, supcriterion, unsupcriterion, optimizer, writer, eps):
    model.train()
    train_loss = 0
    train_acc = 0
    sup_generator = iter(suploader)
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    resize = transforms.Resize((224, 224))
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
        unsup_inputs = resize(normalize(inputs.cuda() / 255.))
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
        
        running_acc =(sup_outputs.argmax(1) == labels).sum().item() 
        train_acc += running_acc
        writer.add_scalar("train acc", running_acc, ep * len(unsuploader) + i)
    train_loss /= len(unsuploader)
    train_acc /= len(unsuploader.dataset)
    return train_loss, train_acc, model, optimizer


def eval(ep, model, loader, criterion, writer, eps):
    model.eval()
    val_loss = 0
    val_acc = 0
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


def save_checkpoint(ep, model, optimizer, scheduler, savepath, isbest):
    save_dict = {
        "epoch": ep,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
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