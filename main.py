
'''
作者：张子昊 2019/11/08
'''
from __future__ import print_function
from __future__ import division
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

#yolo的工具
from models import Darknet
from utils.datasets import ListDataset
from utils.utils import load_classes
from utils.parse_config import parse_data_config,parse_model_config
from train import train
from test import test

#import dataset
'''第一部分写一个输入参数函数'''

'''第二部分写一个构造网络部分'''

'''第三部分写一个数据部分'''

'''第四部分写一个训练部分'''

'''第五部分写一个测试部分'''
#python -u main.py --pretrained_weights  /media/nnir712/F264A15264A119FD/zzh/detect/yolo/PyTorch-YOLOv3/weights/darknet53.conv.74
def create_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--cfg", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data", type=str, default="config/yolov3.data", help="path to data config file")
    parser.add_argument('--batch_size', type=int, default=64,help='input batch size')
    parser.add_argument('--test_batch_size', type=int, default=64,help='test_batch_size')
    parser.add_argument('--epochs', type=int, default=100,help='epochs')
    parser.add_argument('--no_cuda', action='store_true', default=False,help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1,help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100,help='how many batches to wait before logging training status')
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
    
def create_model(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg).to(device)
    #model.apply(weights_init_normal)
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)
    model_cfg = parse_model_config(opt.cfg)
    return model,model_cfg
    
def create_dataset(opt,model_cfg):
    data_config = parse_data_config(opt.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    
    train_loader,valid_loader,test_loader = None,None,None
    
    train_dataset = ListDataset(train_path, img_size=int(model_cfg[0]['width']), augment=True, multiscale=opt.multiscale_training)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    #import pdb;pdb.set_trace()
    valid_dataset = ListDataset(train_path, img_size=int(model_cfg[0]['width']), augment=False, multiscale=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, collate_fn=valid_dataset.collate_fn)    
    test_loader = valid_loader
    return class_names,train_loader,valid_loader,test_loader

def adjust_learning_rate(optimizer,lr, epoch,step=[10,50]):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in step:
        lr *= 0.1 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer,lr
    
def create_optimizer(model,model_cfg):
    #optimizer = torch.optim.Adam(model.parameters())
    #optimizer = optim.SGD(model.parameters(), lr=model.module_defs[0]['learning_rate'], momentum=model.module_defs[0]['momentum'],weight_decay=model.module_defs[0]['decay'])
    lr,momentum,decay = float(model_cfg[0]['learning_rate']),float(model_cfg[0]['momentum']),float(model_cfg[0]['decay'])
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=decay)

    return optimizer,lr
    
class save_model():
    def __init__(self,opt):
        data_config = parse_data_config(opt.data)
        backup = data_config["backup"]
        if not os.path.exists(backup):os.makedirs(backup)
        model_name = os.path.splitext(os.path.split(opt.cfg)[-1])[0]+'_best.pth'
        self.save_path = os.path.join(backup,model_name+'_best.pth')
        self.best_map = 0
    def save(self,mAP):
        if mAP>self.best_map:self.best_map = mAP
        if os.path.exists(self.save_path): os.remove(self.save_path)
        torch.save(model.state_dict(),self.save_path)

if __name__ == "__main__":
    args = create_args()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    model,model_cfg = create_model(args)
    optimizer,lr = create_optimizer(model,model_cfg)
    class_names,train_loader,valid_loader,test_loader = create_dataset(args,model_cfg)
    save = save_model(args)
    #import pdb;pdb.set_trace()
    mAP = 0
    for epoch in range(1, args.epochs + 1):
        train(args,model,train_loader,optimizer,epoch)
        if epoch % 3 == 2 or epoch==args.epochs:
            mAP = test(args,model,model_cfg,test_loader,class_names)
        optimizer,lr = adjust_learning_rate(optimizer,lr, epoch)
        save.save(mAP)
        
    

