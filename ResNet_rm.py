import torchvision.models as models
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from HorizontalMaxPool2D import HorizontalMaxPool2D
from torch.nn import functional as F
from IPython import embed


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)
        

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class resnet(nn.Module):

    def __init__(self, block, layers, num_classes,loss={'softmax'},aligned=False):
        self.inplanes = 64
        super(resnet, self).__init__()
        self.loss = loss
        self.aligned = aligned

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(8,1))
        self.classifier = nn.Linear(2048,num_classes)
        self.horizon_pool = HorizontalMaxPool2D()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(2048,128,kernel_size=1,stride=1,padding=0,bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu1(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv2(lf)
            #lf = self.avgpool(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf/torch.pow(lf,2).sum(dim=1,keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x,x.size()[2:])
        f = x.view(x.size(0),-1)
        
        #f = 1.*f/(torch.norm(f,2,dim=-1,keepdim=True).expand_as(f)+1e-12)
        if not self.training:
            return f,lf
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return f,lf
            return f
        elif self.loss == {'softmax','metric'}:
            if self.aligned: return y,f,lf
            return y,f
        else:
            print('loss selecting error')


    def pre_dist(self):
        pretrained_model = models.resnet50(pretrained=True)
        model = resnet(block = Bottleneck, layers = [3, 4, 6, 3],num_classes=dataset.num_train_pids,aligned=True,loss={'softmax','metric'})
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

#if __name__ == '__main__':
#    model = resnet(block = Bottleneck, layers = [3, 4, 6, 3],num_classes=751,aligned=True,loss={'softmax','metric'})
#    #imgs = torch.Tensor(32,3,384,192)
#    imgs = torch.Tensor(32,3,256,128)
#    y,f,llf = model(imgs)
#    embed()


