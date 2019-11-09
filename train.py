import os
import torch.optim as optim

from utils.parse_config import parse_data_config
from utils.utils import load_classes
import torch
from torch.autograd import Variable
import time
import datetime
def train(opt,model,train_loader,optimizer,epoch=0):
    
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch_i, (_, imgs, targets) in enumerate(train_loader):
        start_time = time.time()
        batches_done = len(train_loader) * epoch + batch_i

        imgs = Variable(imgs.to(device))
        targets = Variable(targets.to(device), requires_grad=False)
        #import pdb;pdb.set_trace()
        optimizer.zero_grad()
        loss, outputs = model(imgs, targets)
        loss.backward()
        optimizer.step()

        # ----------------
        #   Log progress
        # ----------------
        time_left = datetime.timedelta(seconds=time.time() - start_time)
        log_str = "[Epoch %d/%d, Batch %d/%d] loss:%f " % (epoch, opt.epochs, batch_i, len(train_loader)//opt.batch_size,loss)
        log_str += f"{time_left} "
        log_str += "cls_acc:%f "% model.yolo_layers[-1].metrics["cls_acc"]
        log_str += "recall50:%f "% model.yolo_layers[-1].metrics["recall50"]
        if batch_i % 5 == 0:
            print(log_str)

        model.seen += imgs.size(0)
        
    #save
    if epoch % 1 == 0:
        data_config = parse_data_config(opt.data)
        backup = data_config["backup"]
        if not os.path.exists(backup):os.makedirs(backup)
        model_name = os.path.splitext(os.path.split(opt.cfg)[-1])[0]
        save_path = os.path.join(backup,model_name+'_last.pth')
        if os.path.exists(save_path): os.remove(save_path)
        torch.save(model.state_dict(), save_path)

