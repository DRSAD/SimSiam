import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from Utils.lr_scheduler import LR_Scheduler

class linearClassifyTrainer:

    def __init__(self,
                 device,
                 train_dataset,
                 test_dataset):

        self.device=device
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset

    def getOptimizer(self,model,args):
        return optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),weight_decay=args['weight_decay'],
                            momentum=args['momentum'],lr=args['base_lr']*args['batch_size']/256)

    def computeLoss(self,predict,label):
        return F.cross_entropy(predict,label)

    def test(self,model,linear_classifier,test_dataloader):

        correct, total = 0, 0
        for setp, (imgs, labels) in enumerate(test_dataloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = linear_classifier(model(imgs))
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct.item() / total

        return accuracy

    def train(self,model,linear_classifier,args):

        optimizer=self.getOptimizer(linear_classifier,args)

        model.eval()

        train_dataloader = DataLoader(dataset=self.train_dataset,
                                      shuffle=True,
                                      batch_size=args['batch_size'],
                                      num_workers=4)

        test_dataloader = DataLoader(dataset=self.test_dataset,
                                     shuffle=True,
                                     batch_size=args['batch_size'],
                                     num_workers=4)

        lr_scheduler = LR_Scheduler(optimizer=optimizer,
                                    warmup_epochs=args['warmup_epochs'],
                                    warmup_lr=args['warmup_lr'],
                                    num_epochs=args['epochs'],
                                    base_lr=args['base_lr']*args['batch_size']/256,
                                    final_lr=args['final_lr'],
                                    iter_per_epoch=len(train_dataloader),)

        for epoch in range(args['epochs']):
            for step, (imgs, labels) in enumerate(train_dataloader):
                imgs,labels=imgs.to(self.device),labels.to(self.device)

                with torch.no_grad():
                    x=model(imgs)

                predictions=linear_classifier(x)

                loss=self.computeLoss(predictions,labels)

                optimizer.zero_grad()
                loss.backward()
                print("epoch:%d,step:%d,loss:%.5f" % (epoch, step, loss.item()),end='')
                optimizer.step()
                lr=lr_scheduler.step()
                print(' lr:%.9f'%lr)
            train_accuracy = self.test(model,linear_classifier, train_dataloader)
            accuracy = self.test(model,linear_classifier, test_dataloader)
            print("epoch:%d,accuracy is %f,train accuracy:%f" % (epoch, accuracy, train_accuracy))

