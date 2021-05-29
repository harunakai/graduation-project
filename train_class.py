#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable

import data_manager
from data_loader import ImageDataset
import transforms as T
import ResNet
from losses import CrossEntropyLabelSmooth, CenterLoss,CrossEntropyLoss
from utils import AverageMeter, Logger, save_checkpoint
import optimizers
from eval_metrics import eval_market1501

from IPython import embed

parser = argparse.ArgumentParser(description='Train image model with center loss')
# Datasets
parser.add_argument('--root', type=str, default='/home/admin/jupyter', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split_id', type=int, default=0, help="split index")

# Optimization options
parser.add_argument('--labelsmooth',action='store_true',help='label smooth')
parser.add_argument('--optim',type=str,default='adam',help='optimization algorithm(see optimizers.py)')
parser.add_argument('--max_epoch', default=120, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start_epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train_batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test_batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning_rate', default=0.05, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=40, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight_decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-4)")
# Architecture
#parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
# Miscs
parser.add_argument('--print_freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval_step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start_eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save_dir', type=str, default='log')
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu_devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

args = parser.parse_args()

epochs = args.max_epoch
version =  torch.__version__
dataset = data_manager.init_img_dataset(root=args.root,name=args.dataset)

str_ms = args.ms.split(',')  
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()
    model = ResNet.PCB_test(model)

    model.eval()
    if use_gpu:
        model = model.cuda()
    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()
            
            end = time.time()
            features = extract_feature(model,queryloader)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()
            
            end = time.time()
            features = extract_feature(model,galleryloader)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    
    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf,2).sum(dim=1,keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids,max_rank=20)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]


def main():
    use_gpu =torch.cuda.is_available()
    if args.use_cpu: 
        use_gpu = False
    pin_memory = True if use_gpu else False
    
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        cudnn.benchmark = True
        #torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")
        
    
    transform_train = T.Compose([
        T.Resize((args.height,args.width), interpolation=3),
        T.Pad(10),
        T.RandomCrop((args.height,args.width)),
        #T.Random2DTranslation(args.height,args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])
    
    transform_test = T.Compose([
        T.Resize((args.height,args.width),interpolation=2),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])
    
    train_loader = DataLoader(
        ImageDataset(dataset.train,transform = transform_train),
        batch_size=args.train_batch,num_workers=args.workers,
        shuffle=True,
        pin_memory=pin_memory,drop_last=True,
    )
    query_loader = DataLoader(
        ImageDataset(dataset.query,transform = transform_test),
        batch_size=args.test_batch,num_workers=args.workers,
        shuffle=False,
        pin_memory=pin_memory,drop_last=False,
    )
    gallery_loader = DataLoader(
        ImageDataset(dataset.gallery,transform = transform_test),
        batch_size=args.test_batch,num_workers=args.workers,
        shuffle=False,
        pin_memory=pin_memory,drop_last=False,
    )
    print("Initializing model: {}".format('resnet50'))
    model = ResNet.PCB(num_classes=dataset.num_train_pids)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    
    #criterion_class = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    criterion_class = nn.CrossEntropyLoss()
    optimizer = optimizers.init_optim(args.optim,model.parameters(),args.lr,args.weight_decay)
    
    
    

    if args.stepsize >0:
        scheduler = lr_scheduler.StepLR(optimizer,step_size=args.stepsize,gamma=args.gamma)
    
    start_epoch = args.start_epoch
    
    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
    
    if use_gpu:
        model = model.cuda()

    if args.evaluate:
        print('Evaluate Only!')
        test(model,query_loader,gallery_loader,use_gpu)
        return 0
        
    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print(model)
    print("==> Start training")    
    
    for epoch in range(start_epoch,epochs):
        start_train_time = time.time()
        train(epoch,model,criterion_class,optimizer,train_loader,use_gpu)
        train_time += round(time.time()-start_train_time)
        
        if args.stepsize> 0: scheduler.step()
        
        
    if (epoch+1)>args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == epochs:
        print("==> Test")
        model = model.eval()
        rank1 = test(model, query_loader, gallery_loader, use_gpu)        
        
    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))
        
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
        
def train(epoch,model,criterion_class,optimizer,train_loader,use_gpu):
    model.train()
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    running_loss=0.0
    running_corrects =0.0

    
    end = time.time()
    for batch_idx, (imgs,pids,_) in enumerate(train_loader):
        now_batch_size,c,h,w = imgs.shape
        if use_gpu:
            imgs,pids =  Variable(imgs.cuda().detach()), Variable(pids.cuda().detach())
            
        # measure data loading time
        data_time.update(time.time() - end)
        
        output = model(imgs)
        part = {}
        num_part = 6
        sm = nn.Softmax(dim=1)
        for i in range(num_part):
            part[i]=output[i]

        score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
        _, preds = torch.max(score.data, 1)

        loss = criterion_class(part[0], pids)
        for i in range(num_part-1):
            loss += criterion_class(part[i+1], pids)
        loss.backward()
        optimizer.step()
        if int(version[0])>0 or int(version[2]) > 3:
            running_loss += loss.item() * now_batch_size
        else:
            running_loss += loss.data[0] * now_batch_size
        running_corrects += float(torch.sum(preds == pids.data))
        epoch_loss = running_loss / dataset.num_train_pids
        epoch_acc = running_corrects / dataset.num_train_pids

        batch_time.update(time.time()-end)
        end=time.time()
        losses.update(loss.item(),pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            #draw_curve(epoch)
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch+1, batch_idx+1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
        #break
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for batch_idx, (imgs,pids,_) in enumerate(dataloaders):
        img, label = imgs,pids
        n, c, h, w = img.size()
        count += n
        print(count)

        ff = torch.FloatTensor(n,2048,6).zero_().cuda() 
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        # norm feature
        
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)

        features = torch.cat((features,ff.data.cpu()), 0)
    return features
        

if __name__ == '__main__':
    main()


# In[ ]:




