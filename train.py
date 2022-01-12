from tqdm import tqdm
import torch
from test import test
import os
class Train:
    def __init__(self, model,train_loader,test_loader,lr,epochs,root_dir,ckpt_dir):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.root_dir = root_dir
        self.ckpt_dir = ckpt_dir
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
        print(next(model.parameters()).device)
        os.makedirs(os.path.join(ckpt_dir, self.model.name), exist_ok=True)

        '''criterion'''
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        '''optimizer'''
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        '''scheduler'''
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)

    # def calc_acc(preds, labels):
    #       _, pred_max = torch.max(preds, 1)
    #       acc = torch.sum(pred_max == labels.data, dtype=torch.float64) / len(preds)
    #       return acc

    def train_(self):
        total_train_loss = 0.0
        for epoch in range(self.epochs):
            self.model.train(True)
            train_loss = 0.0
            acc = 0.0
            for image, lable in tqdm(self.train_loader):
                image = image.to(self.device)
                lable = lable.to(self.device)
                predict = self.model(image)

                loss = self.criterion(predict, lable)
                loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()
                train_loss += loss
                # acc += self.calc_acc(predict, lable)
                _, pred_max = torch.max(predict, 1)
                acc += torch.sum(pred_max == lable.data, dtype=torch.float64) / len(predict)

            epoch_loss = train_loss / len(self.train_loader)
            epoch_acc = acc / len(self.train_loader)
            self.scheduler.step(epoch_loss)
            print('epoch:{} | train_loss: {:.4f}, train_accuracy :{:.2f}'.format(epoch + 1, epoch_loss, epoch_acc))

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(),
                           os.path.join(self.ckpt_dir, self.model.name, f'ckpt_{epoch}.pth'))

            if epoch % 5 == 0:
                self.test(self.model,self.test_loader)

