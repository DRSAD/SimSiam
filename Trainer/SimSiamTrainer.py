import torch.optim as optim
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from Utils.lr_scheduler import LR_Scheduler

class simSiamTrainer:

    def __init__(self,
                 device,
                 train_dataset):

        self.device = device
        self.train_dataset = train_dataset


    def getOptimizer(self,model,args):
        return optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=args['weight_decay'],
                         momentum=args['momentum'], lr=args['base_lr'] * args['batch_size'] / 256)

    def computeLoss(self,p1,z1):
        return -F.cosine_similarity(p1,z1).mean()

    def train(self,model,args):

        optimizer=self.getOptimizer(model,args)

        train_dataloader = DataLoader(dataset=self.train_dataset,
                                      shuffle=True,
                                      batch_size=args['batch_size'],
                                      num_workers=4)

        lr_scheduler = LR_Scheduler(optimizer=optimizer,
                                    warmup_epochs=args['warmup_epochs'],
                                    warmup_lr=args['warmup_lr'],
                                    num_epochs=args['epochs'],
                                    base_lr=args['base_lr'] * args['batch_size'] / 256,
                                    final_lr=args['final_lr'],
                                    iter_per_epoch=len(train_dataloader), )
        model.train()

        for epoch in range(args['epochs']):
            for step, ((imgs1, imgs2), labels) in enumerate(train_dataloader):
                imgs1,imgs2,labels=imgs1.to(self.device),imgs2.to(self.device),labels.to(self.device)

                p1,z2=model(imgs1,imgs2)
                p2,z1=model(imgs2,imgs1)

                loss=self.computeLoss(p1,z2)/2+self.computeLoss(p2,z1)/2

                optimizer.zero_grad()
                loss.backward()
                print("epoch:%d,step:%d,loss:%.5f" % (epoch, step, loss.item()),end='')
                optimizer.step()
                lr=lr_scheduler.step()
                print(' lr:%.9f'%lr)
                #break
            #break




