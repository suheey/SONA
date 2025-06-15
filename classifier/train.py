from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import os
import pickle
import argparse
import time
import threading
from functools import partial
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm as original_tqdm
from accelerate import Accelerator

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F



from model.network import ResNet18_224x224
import gc


parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='in200', choices=['in200'],
                    help='Choose between in200, in30, in100')

parser.add_argument('--id_train_dir', type=str, default='/lab-di/lab-di/squads/ood_detection/dataset/ImageNet-200/train', help='ID train_dir')
parser.add_argument('--id_train_vae_dir', type=str, default='/lab-di/lab-di/squads/ood_detection/dataset/ImageNet-200/train_vae', help='ID train_dir')

parser.add_argument('--id_test_dir', type=str, default='/lab-di/lab-di/squads/ood_detection/dataset/ImageNet-200/val', help='ID test_dir')
parser.add_argument('--ood_dir', type=str, default='/lab-di/lab-di/squads/ood_detection/dataset/ImageNet-200/sd_outlier/og_sega_nuis_111_0.2_w30', help='OOD train_dir')

parser.add_argument('--model', '-m', type=str, default='resnet18',
                    choices=['resnet18'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./ckpts/ledits_1525_30', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
# parser.add_argument('--ngpu', type=int, default=8, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=8, help='Pre-fetching threads.')
args = parser.parse_args()

if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)
print("ood_dir: ", args.ood_dir)
accelerator = Accelerator()
_is_local_main_process = accelerator.is_local_main_process
tqdm = partial(original_tqdm, disable=not _is_local_main_process, position=0)
state = {k: v for k, v in args._get_kwargs()}
if _is_local_main_process:
    print(state)

torch.manual_seed(1)
np.random.seed(1)

def load_dataset(id_train_dir, id_train_vae_dir, id_test_dir, ood_dir):
    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

    # ImageFolder를 감싸서 예외 처리를 해주는 클래스 정의
    class SafeImageFolder(torchvision.datasets.ImageFolder):
        def __getitem__(self, index):
            try:
                return super().__getitem__(index)
            except UnidentifiedImageError:
                print(f"Cannot identify image file: {self.samples[index][0]}")
                return None

    train_data_in = SafeImageFolder(
        os.path.join(id_train_dir),
        trn.Compose([
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ]))
    train_data_in_vae = SafeImageFolder(
        os.path.join(id_train_vae_dir),
        trn.Compose([
            trn.RandomResizedCrop(224),
            trn.RandomHorizontalFlip(),
            trn.ToTensor(),
            normalize,
        ]))
    combined_train_data = data.ConcatDataset([d for d in [train_data_in, train_data_in_vae] if d is not None])
    train_data_out = SafeImageFolder(
        ood_dir,
        trn.Compose([
        trn.RandomResizedCrop(224),
        trn.RandomHorizontalFlip(),
        trn.ToTensor(),
        normalize,]))
    test_data = SafeImageFolder(
        os.path.join(id_test_dir),
        trn.Compose([
            trn.Resize(256),
            trn.CenterCrop(224),
            trn.ToTensor(),
            normalize,
        ]))

    trainloader_in = torch.utils.data.DataLoader(
        combined_train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    trainloader_in_vae = torch.utils.data.DataLoader(
        train_data_in_vae,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    trainloader_out = torch.utils.data.DataLoader(
        train_data_out,
        batch_size=args.oe_batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.prefetch, pin_memory=True)

    return trainloader_in, trainloader_out, testloader


class Accuracy:
    """Accuracy score."""
    def __init__(self):
        super().__init__()
        self.__build()

    def __build(self):
        self._lock = threading.Lock()
        self._predictions = []
        self._targets = []

    def reset(self):
        self._predictions.clear()
        self._targets.clear()

    def update(self, output):
        y_pred, y_true = output
        with self._lock:
            self._predictions.append(y_pred)
            self._targets.append(y_true)

    def compute(self):
        with self._lock:
            predictions = torch.cat(self._predictions, dim=0).numpy()
            targets = torch.cat(self._targets, dim=0).numpy()
            return accuracy_score(y_true=targets, y_pred=predictions)

### 1) data
train_loader_in, train_loader_out, test_loader = load_dataset(args.id_train_dir, args.id_train_vae_dir, args.id_test_dir, args.ood_dir)

if args.dataset == 'in200':
    num_classes=200

### 2) model
net = ResNet18_224x224(num_classes=200)
# pretrained_weights = torch.load('./ckpts/temp.pt')
# net.load_state_dict(pretrained_weights, strict=True)
# net.cuda()

start_epoch = 0
# Restore model if desired => not use
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = os.path.join(args.load, args.dataset  + '_' + args.model + '_oe_scratch_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"


### 3) optimizer
optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)

### 4) scheduler
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))

### accelerate setting
train_loader_in, train_loader_out, test_loader, net, optimizer = accelerator.prepare(
    train_loader_in, train_loader_out, test_loader, net, optimizer
)

def compute_mutual_information_kl(x, y):
    batch_size = x.size(0)

    joint = torch.bmm(x.unsqueeze(2), y.unsqueeze(1))  
    joint = joint.view(batch_size, -1)
    joint = F.softmax(joint, dim=1)

    marginal_x = x.mean(0, keepdim=True)
    marginal_y = y.mean(0, keepdim=True)
    marginal = torch.mm(marginal_x.t(), marginal_y).view(1, -1)
    marginal = F.softmax(marginal, dim=1)
    
    mi = F.kl_div(joint.log(), marginal.repeat(batch_size, 1), reduction='batchmean')
    return mi


def compute_mutual_information(x, y, eps=1e-8):
    batch_size, dim = x.size()

    # 결합 분포 추정
    joint = torch.bmm(x.unsqueeze(2), y.unsqueeze(1))  # (batch_size, dim, dim)
    joint = joint.view(batch_size, -1)
    joint = joint / joint.sum(dim=1, keepdim=True)
    joint = joint.mean(dim=0)  # 배치에 대해 평균

    # 주변 분포 추정
    p_x = x.mean(dim=0)
    p_y = y.mean(dim=0)
    prod = torch.mm(p_x.unsqueeze(1), p_y.unsqueeze(0)).view(-1)

    # 상호 정보량 계산
    mi = torch.sum(joint * torch.log(joint / (prod + eps) + eps))
    return mi

def train():
    net.train()

    batch_iterator = iter(train_loader_out)
    total_steps = min(len(train_loader_in), len(train_loader_out))
    with tqdm(total=total_steps, desc=f'Epoch {epoch:03d}', position=0) as progress_bar:
        for in_set in train_loader_in:
            try:
                out_set = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(train_loader_out)
                out_set = next(batch_iterator)

            data = torch.cat((in_set[0], out_set[0]), 0)
            target = in_set[1]
            data = data.to(accelerator.device)
            target = target.to(accelerator.device)

            optimizer.zero_grad()

            x, features = net(data, return_feature=True)
            loss = F.cross_entropy(x[:len(in_set[0])], target)

            loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()


            id_features = features[:len(in_set[0])]
            ood_features = features[len(in_set[0]):]

            #loss += 0.7 * compute_mutual_information_kl(id_features, ood_features)
            loss += 0.7 * compute_mutual_information(id_features, ood_features)


            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            torch.cuda.synchronize()
            progress_bar.update(1)

        # 가비지 컬렉션 수행
        gc.collect()


def evaluate():
    net.eval()
    test_losses = []
    test_accuracy = Accuracy()
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=f'eval (epoch {epoch:03d})'):
            data, target = data.to(accelerator.device), target.to(accelerator.device)
            output = net(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            test_losses.append(accelerator.gather(loss))
            preds = output.argmax(dim=1, keepdim=True)
            test_accuracy.update((accelerator.gather(preds).detach().cpu(),
                                  accelerator.gather(target).detach().cpu()))

    test_loss = torch.sum(torch.cat(test_losses)) / len(test_loader.dataset)
    test_acc = test_accuracy.compute()
    test_accuracy.reset()
    return test_acc, test_loss



start_time = time.time()
for epoch in range(start_epoch, args.epochs):
    if _is_local_main_process:
        print(f"Current LR: {scheduler.get_lr()[0]:.3f}")

    train()
    if epoch in [0, 30, 50] or epoch > 79:
        eval_accuracy, eval_loss = evaluate()

        if _is_local_main_process:
            print(f'Epoch {epoch:03d} / Eval_loss = {eval_loss.item():.3f} Eval accuracy = {eval_accuracy:.3f}')

        if _is_local_main_process and args.save is not None:
            unwrapped_model = accelerator.unwrap_model(net)
            torch.save(unwrapped_model.state_dict(), os.path.join(args.save, f'epoch_{epoch}.pt'))


end_time = time.time()
training_time = end_time - start_time
hours = int(training_time // 3600)
minutes = int((training_time % 3600) // 60)
seconds = int(training_time % 60)
if _is_local_main_process:
    print(f"Training_time: {hours}:{minutes}:{seconds}")