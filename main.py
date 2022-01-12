import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from train import Train
from mymodel import MyModel
from argparse import ArgumentParser
import os

transform = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    # torchvision.transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.Normalize(mean=0.5,std=0.5)
])


parser = ArgumentParser()
parser.add_argument("--root_dir", default='drive/MyDrive/')
parser.add_argument("--ckpt_dir", default=os.path.join('drive/MyDrive/', 'checkpoints'))
parser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
parser.add_argument("--batch_size", help="batch size", type=int, default=1)
parser.add_argument("--epochs", help="number of epochs", type=int, default=50)
parser.add_argument("--numworker", help="number of threads", type=int, default=0)
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
args = parser.parse_args()

lr = args.lr
epochs = args.epochs
batchsize = args.batch_size
numworker = args.numworker
root_dir = args.root_dir
ckpt_dir = args.ckpt_dir

train_data = FashionMNIST('dataset',train=True,download=True,transform = transform)
test_data = FashionMNIST('dataset',train=False,download=True,transform = transform)
train_loader = DataLoader(train_data,batch_size=batchsize, shuffle=True,num_workers=numworker)
test_loader = DataLoader(train_data,batch_size=batchsize, shuffle=True,num_workers=numworker)

model = MyModel()
train = Train(model,train_loader,test_loader,lr,epochs,root_dir,ckpt_dir)
train.train_()