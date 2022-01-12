from tqdm import tqdm
import torch

def test(self,model,test_loader):
    for image, lable in tqdm(test_loader):
        self.model.eval()
        test_loss = 0.0
        acc = 0.0
        image = image.to(self.device)
        lable = lable.to(self.device)
        predict = self.model(image)
        loss = self.criterion(predict, lable)

        test_loss += loss
        # acc += self.calc_acc(predict, lable)
        _, pred_max = torch.max(predict, 1)
        acc += torch.sum(pred_max == lable.data, dtype=torch.float64) / len(predict)

    epoch_loss = test_loss / len(test_loader)
    epoch_acc = acc / len(test_loader)
    print(' test_loss: {:.4f}, test_accuracy:{:.2f}'.format(epoch_loss, epoch_acc))



