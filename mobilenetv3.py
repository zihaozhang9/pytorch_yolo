'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

#yolo
from utils.utils import build_targets, to_cpu
from models import YOLOLayer

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )


        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out



class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )


        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out



def test():
    net = MobileNetV3_Small()
    x = torch.randn(2,3,224,224)
    y = net(x)
    print(y.size())

# test()


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x

class conv_black(nn.Module):
    def __init__(self, in_size,kernel_size,out_size):
        super(conv_black, self).__init__()
        pad = (kernel_size - 1) // 2
        self.conv =  nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=1, padding=pad, bias=False),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(0.1, inplace=True),
            )
        #self.cv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        #self.bn = nn.BatchNorm2d(out_size)
        #self.lr = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.conv(x)
        #x = self.cv(x)
        #x = self.bn(x)
        #import pdb;pdb.set_trace()
        #x = self.lr(x)
        return x
        
class MobileNetV3_Large_yolo(nn.Module):
    #参考：yolo v3 https://blog.csdn.net/qq_37541097/article/details/81214953
    #参考：mobilenet v3 https://blog.csdn.net/thisiszdy/article/details/90167304
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416,num_anchors=3,num_classes=4):
        super(MobileNetV3_Large_yolo, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)#208
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        
        self.bneck1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),#208
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),#104
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),#104
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),#52
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1)#52
        )
        
        self.bneck2 = nn.Sequential(
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),#52
            Block(3, 40, 240, 80, hswish(), None, 2),#26
            Block(3, 80, 200, 80, hswish(), None, 1),#26
            Block(3, 80, 184, 80, hswish(), None, 1),#26
            Block(3, 80, 184, 80, hswish(), None, 1),#26
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),#26
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),#26
        )
        
        self.bneck3 = nn.Sequential(
            Block(5, 112, 672, 160, hswish(), SeModule(160), 2),#13
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),#13
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),#13
        )
        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        #--------------------------------
        #上面是 mobilelarge 的骨干网
        #--------------------------------
        self.conv_set1 = nn.Sequential(
            conv_black(960,1,512),
            conv_black(512,3,1024),
            conv_black(1024,1,512),
            conv_black(512,3,1024),
            conv_black(1024,1,512)
        )
        #32次下采样输出卷积
        self.conv_out1 = nn.Sequential(
            conv_black(512,3,1024),
            nn.Conv2d(1024, num_anchors*(num_classes+5), kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.anchors1 = [(116,90),  (156,198),  (373,326)]#[(10,13),  (16,30),  (33,23)]
        self.yolo_layer1 = YOLOLayer(self.anchors1, num_classes, img_size)
        self.upsample1 = nn.Sequential( 
            conv_black(512,1,256),
            Upsample(scale_factor=2, mode="nearest")
        )
        
        
        self.conv_set2 = nn.Sequential(
            conv_black(368,1,256),
            conv_black(256,3,512),
            conv_black(512,1,256),
            conv_black(256,3,512),
            conv_black(512,1,256)
        )
        
        #16次下采样输出卷积
        self.conv_out2 = nn.Sequential(
            conv_black(256,3,512),
            nn.Conv2d(512, num_anchors*(num_classes+5), kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.anchors2 = [(30,61),  (62,45),  (59,119)]
        self.yolo_layer2 = YOLOLayer(self.anchors2, num_classes, img_size)
        self.upsample2 = nn.Sequential( 
            conv_black(256,1,128),
            Upsample(scale_factor=2, mode="nearest")
        )
        
        self.conv_set3 = nn.Sequential(
            conv_black(168,1,128),
            conv_black(128,3,256),
            conv_black(256,1,128),
            conv_black(128,3,256),
            conv_black(256,1,128)
        )
        
        #8次下采样输出卷积
        self.conv_out3 = nn.Sequential(
            conv_black(128,3,256),
            nn.Conv2d(256, num_anchors*(num_classes+5), kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.anchors3 = [(10,13),  (16,30),  (33,23)]#[(116,90),  (156,198),  (373,326)]
        self.yolo_layer3 = YOLOLayer(self.anchors3, num_classes, img_size)
        self.yolo_layers = [self.yolo_layer1,self.yolo_layer2,self.yolo_layer3]
        self.seen = 0

    def forward(self, x, targets=None):
        #import pdb;pdb.set_trace()
        img_dim = x.shape[2]
        loss = 0
        yolo_outputs = []
        #mobilenet v3 骨干网络
        out = self.hs1(self.bn1(self.conv1(x)))
        bneck1_out = self.bneck1(out)
        bneck2_out = self.bneck2(bneck1_out)
        bneck3_out = self.bneck3(bneck2_out)
        bneck3_out = self.hs2( self.bn2(self.conv2(bneck3_out)) ) #13
        
        #conv set
        conv_set1_out = self.conv_set1(bneck3_out)
        
        #32次下采样输出层
        out = self.conv_out1(conv_set1_out)
        x, layer_loss = self.yolo_layer1(out, targets, img_dim)
        loss += layer_loss
        yolo_outputs.append(x)
        
        #上采样
        out = self.upsample1(conv_set1_out)
        #import pdb;pdb.set_trace()
        cat_out1 = torch.cat((out, bneck2_out), 1)#512+112 = 624
        
        #conv set
        #import pdb;pdb.set_trace()
        conv_set2_out = self.conv_set2(cat_out1)
        
        #16次下采样输出层
        out = self.conv_out2(conv_set2_out)
        x, layer_loss = self.yolo_layer2(out, targets, img_dim)
        loss += layer_loss
        yolo_outputs.append(x)
        
        #上采样
        #out = self.conv_set2(bneck3_out)
        #import pdb;pdb.set_trace()
        out = self.upsample2(conv_set2_out)
        cat_out2 = torch.cat((out, bneck1_out), 1)#128+40 =168
        
        
        conv_set3_out = self.conv_set3(cat_out2) #in 168
        
        #8次下采样输出层
        out = self.conv_out3(conv_set3_out)
        x, layer_loss = self.yolo_layer3(out, targets, img_dim)
        loss += layer_loss
        yolo_outputs.append(x)
        
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)
        
class MobileNetV3_Small_yolo(nn.Module):
    #参考：yolo v3 https://blog.csdn.net/qq_37541097/article/details/81214953
    #参考：mobilenet v3 https://blog.csdn.net/thisiszdy/article/details/90167304
    """YOLOv3 object detection model"""

    def __init__(self, config_path, img_size=416,num_anchors=3,num_classes=4):
        super(MobileNetV3_Small_yolo, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)#208
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        
        self.bneck1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),#112 #104
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),#56 #
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),#28
        )
        
        self.bneck2 = nn.Sequential(
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),#28
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),#14
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),#14
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),#14
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),#14
        )
        
        self.bneck3 = nn.Sequential(
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),#14
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),#7
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),#7
        )
        # 
        #self.conv_set1 = nn.Sequential(
        #    nn.Conv2d(96, 512, kernel_size=3, stride=1, padding=1, bias=False),
        #    nn.BatchNorm2d(512)
        #)
        self.conv1_1 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        #concate 256+48 = 
        #self.conv_set2 = nn.Sequential(
        #    nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #    nn.BatchNorm2d(256),
        #    nn.Conv2d(256, num_anchors*(num_classes+5), kernel_size=1, stride=1, padding=0, bias=False)
        #)
        self.conv1_2 = nn.Conv2d(144, 144, kernel_size=3, stride=1, padding=1, bias=False)
             
        #32次下采样输出卷积
        self.conv_out1 = nn.Sequential(
            nn.Conv2d(96, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, num_anchors*(num_classes+5), kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.anchors1 = [(116,90),  (156,198),  (373,326)]#[(10,13),  (16,30),  (33,23)]
        self.yolo_layer1 = YOLOLayer(self.anchors1, num_classes, img_size)
        self.upsample1 = Upsample(scale_factor=2, mode="nearest")
        
        #16次下采样输出卷积
        self.conv_out2 = nn.Sequential(
            nn.Conv2d(144, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, num_anchors*(num_classes+5), kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.anchors2 = [(30,61),  (62,45),  (59,119)]
        self.yolo_layer2 = YOLOLayer(self.anchors2, num_classes, img_size)
        self.upsample2 = Upsample(scale_factor=2, mode="nearest")
        
        #8次下采样输出卷积
        self.conv_out3 = nn.Sequential(
            nn.Conv2d(168, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, num_anchors*(num_classes+5), kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.anchors3 = [(10,13),  (16,30),  (33,23)]#[(116,90),  (156,198),  (373,326)]
        self.yolo_layer3 = YOLOLayer(self.anchors3, num_classes, img_size)
        self.yolo_layers = [self.yolo_layer1,self.yolo_layer2,self.yolo_layer3]
        self.seen = 0

    def forward(self, x, targets=None):
        #import pdb;pdb.set_trace()
        img_dim = x.shape[2]
        loss = 0
        yolo_outputs = []
        #mobilenet v3 骨干网络
        out = self.hs1(self.bn1(self.conv1(x)))
        bneck1_out = self.bneck1(out)
        bneck2_out = self.bneck2(bneck1_out)
        bneck3_out = self.bneck3(bneck2_out)
        
        #上采样
        #out = self.conv_set1(bneck3_out)
        out = self.conv1_1(bneck3_out)
        out = self.upsample1(out)
        cat_out1 = torch.cat((out, bneck2_out), 1)#96+48=144
        
        #上采样
        #out = self.conv_set2(bneck3_out)
        #import pdb;pdb.set_trace()
        out = self.conv1_2(cat_out1)
        out = self.upsample2(out)
        cat_out2 = torch.cat((out, bneck1_out), 1)#96+48+24
        
        #32次下采样输出层
        out = self.conv_out1(bneck3_out)#
        x, layer_loss = self.yolo_layer1(out, targets, img_dim)
        loss += layer_loss
        yolo_outputs.append(x)
        
        #16次下采样输出层
        out = self.conv_out2(cat_out1)
        x, layer_loss = self.yolo_layer2(out, targets, img_dim)
        loss += layer_loss
        yolo_outputs.append(x)
        
        #8次下采样输出层
        out = self.conv_out3(cat_out2)
        x, layer_loss = self.yolo_layer3(out, targets, img_dim)
        loss += layer_loss
        yolo_outputs.append(x)
        
        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))
        return yolo_outputs if targets is None else (loss, yolo_outputs)
        
    
    
