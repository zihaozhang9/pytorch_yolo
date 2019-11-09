import numpy as np
import torch
from torch.autograd import Variable

from terminaltables import AsciiTable
from utils.utils import non_max_suppression,get_batch_statistics,ap_per_class,xywh2xyxy
import tqdm

def evaluate(args,model,model_cfg, test_loader, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5):
    img_size = int(model_cfg[0]['width'])
    model.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    labels ,sample_metrics = [], []
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(test_loader, desc="Detecting objects")):
        #if batch_i >10:break
        labels += targets[:, 1].tolist()
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        #import pdb;pdb.set_trace()
        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        with torch.no_grad():
            #import pdb;pdb.set_trace()
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class
    
def test(args,model,model_cfg, test_loader,class_names):
    print("\n---- test Model ----")
    precision, recall, AP, f1, ap_class = evaluate(args,model,model_cfg,test_loader)
    ap_table = [["Index", "Class name", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    print(AsciiTable(ap_table).table)
    print(f"----Test mAP {AP.mean()} precision {precision.mean()} recall {recall.mean()} f1 {f1.mean()}")
    return AP.mean()

