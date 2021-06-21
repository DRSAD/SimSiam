from Trainer.SimSiamTrainer import simSiamTrainer
from Trainer.LinearClassifyTrainer import linearClassifyTrainer
import torch
import torch.nn as nn

from Utils.DataSetManager import get_dataset

from Models.ResNet import resnet18
from Models.SimSiam import SimSiamModel

import yaml

yamlPath = 'Config/SimSiam_cifar.yaml'
dataset_name='CIFAR10'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_dataset,test_dataset=get_dataset('CIFAR10','/data/zhuoyun/dataset')
SSL_train_dataset,SSL_test_dataset=get_dataset('CIFAR10_SSL','/data/zhuoyun/dataset')

model=SimSiamModel(resnet18()).to(device)

with open(yamlPath,'rb') as f:
    data = list(yaml.safe_load_all(f))
    train_config=data[0]['train']
    eval_config=data[0]['eval']

simsiam_trainer=simSiamTrainer(device,SSL_train_dataset)

linear_classify_trainer=linearClassifyTrainer(device,train_dataset,test_dataset)

simsiam_trainer.train(model,train_config)

#model=torch.load('model/SSL_model.pkl', map_location='cpu').to(device)

linear_classifier=nn.Linear(eval_config['linear_classifier_input_dim'],eval_config['numclass']).to(device)

torch.save(model, 'model/SSL_model.pkl')

linear_classify_trainer.train(model.featureExactor,linear_classifier,eval_config)

torch.save(linear_classifier, 'model/linear_model.pkl')








