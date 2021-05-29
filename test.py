  
import torch.nn as nn
import torch
#from FeatureExtractor import FeatureExtractor
from torchvision import transforms
from IPython import embed
import ResNet
from scipy.spatial.distance import cosine, euclidean
from utils import *
from sklearn.preprocessing import normalize
from IPython import embed

class FeatureExtractor(nn.Module):
    def __init__(self,submodule,extracted_layers):
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module1 in self.submodule._modules.items():
            #embed()
            #if name == "classfier":
            #    x = x.view(x.size(0),-1)
            if "layer" in name:
                
                for block_name, cnn_block in module1._modules.items():
                    x = cnn_block(x)
                    if block_name in self.extracted_layers:
                        outputs.append(x)
                    embed()
            else:
                x = module1(x)
        return outputs
    
    
if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    #model = ResNet.ResNet50(num_classes=751,aligned=True,loss={'softmax','metric'})
    img_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #a = ft_net1(block = Bottleneck, layers = [3, 4, 6, 3])
    model = ResNet.CNN(block = ResNet.Bottleneck, layers = [3, 4, 6, 3],num_classes=751,aligned=True,loss={'softmax','metric'})
    #c = ft_net2(block = Bottleneck, layers = [3, 4, 6, 3])
    imgs = torch.Tensor(32,3,256,128)
    img = torch.Tensor(256,128)
    #imgs = torch.Tensor(32,3,256,128)
    #y,f,llf = model(imgs)
    #n = PCB(751)
    print(model)
    exact_list = ['layer4']
    myexactor = FeatureExtractor(model, exact_list)
    img_path1 = '0005_c1s1_001351_00.jpg'
    img_path2 = '0005_c2s1_000976_00.jpg'
    img1 = read_image(img_path1)
    img2 = read_image(img_path2)
    img1 = img_to_tensor(img1, img_transform)
    img2 = img_to_tensor(img2, img_transform)
    #embed()
    if use_gpu:
        model = model.cuda()
        img1 = img1.cuda()
        img2 = img2.cuda()
    model.eval()
    f1 = myexactor(img1)
    f2 = myexactor(img2)
    #embed()