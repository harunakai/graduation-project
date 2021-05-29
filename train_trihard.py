import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import optimizers
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler


import data_manager
from data_loader import ImageDataset
import transforms as T
import ResNet
from losses import CrossEntropyLabelSmooth, CenterLoss,TripletLoss,CrossEntropyLoss
from sampler import RandomIdentitySampler
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import eval_market1501

from IPython import embed

parser = argparse.ArgumentParser(description='Train image model with center loss')
# Datasets
parser.add_argument('--root', type=str, default='/data-tmp/', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split_id', type=int, default=0, help="split index")

parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")

# Optimization options
parser.add_argument('--labelsmooth',action='store_true',help='label smooth')
parser.add_argument('--optim',type=str,default='adam',help='optimization algorithm(see optimizers.py)')
parser.add_argument('--max_epoch', default=150, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start_epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train_batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test_batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning_rate', default=0.0002, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=50, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight_decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
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

args = parser.parse_args()

epochs = args.max_epoch


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
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")
        
    dataset = data_manager.init_img_dataset(root=args.root,name=args.dataset)
    transform_train = T.Compose([
        T.Random2DTranslation(args.height,args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.48556,0.406],std=[0.229,0.22924,0.225]),
    ])
    
    transform_test = T.Compose([
        T.Resize((args.height,args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.48556,0.406],std=[0.229,0.22924,0.225]),
    ])
    
    train_loader = DataLoader(
        ImageDataset(dataset.train,transform = transform_train),
        sampler=RandomIdentitySampler(dataset.train,num_instances= args.num_instances),
        batch_size=args.train_batch,num_workers=args.workers,
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
    model = ResNet.ResNet50(num_classes=dataset.num_train_pids,loss={'softmax','metric'})
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    
    criterion_class = nn.CrossEntropyLoss()
    criterion_metric = TripletLoss(margin=args.margin)
    optimizer = optimizers.init_optim(args.optim,model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    
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
        train(epoch,model,criterion_class,criterion_metric,optimizer,train_loader,use_gpu)
        train_time += round(time.time()-start_train_time)
        
        if args.stepsize>0:scheduler.step()
        
        if (epoch-1)>args.start_eval and args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == epochs:
            print("==> Test")
            rank1 = test(model, query_loader, gallery_loader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))
        
    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))
        
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
        
def train(epoch,model,criterion_class,criterion_metric,optimizer,train_loader,use_gpu):
    model.train()
    
    losses = AverageMeter()
    xent_losses = AverageMeter()
    triplet_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    
    end = time.time()
    for batch_idx, (imgs,pids,_) in enumerate(train_loader):
        if use_gpu:
            imgs,pids = imgs.cuda(),pids.cuda()
            
        # measure data loading time
        data_time.update(time.time() - end)
        
        outputs, features = model(imgs)
        xent_loss = criterion_class(outputs,pids)
        triplet_loss = criterion_metric(features,pids)
        loss = xent_loss+triplet_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time()-end)
        end=time.time()
        losses.update(loss.item(),pids.size(0))
        xent_losses.update(xent_loss.item(),pids.size(0))
        triplet_losses.update(triplet_loss.item(),pids.size(0))
        
        if (batch_idx+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'CLoss {xent_loss.val:.4f} ({xent_loss.avg:.4f})\t'
                  'TLoss {triplet_loss.val:.4f} ({triplet_loss.avg:.4f})\t'.format(
                   epoch+1, batch_idx+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,xent_loss=xent_losses,triplet_loss=triplet_losses))
        #break
        
        
    return 0


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()
            
            end = time.time()
            features = model(imgs)
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
            features = model(imgs)
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
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(beta=1, alpha=-2, mat1=qf, mat2=gf.t())
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

if __name__ == '__main__':
    main()

