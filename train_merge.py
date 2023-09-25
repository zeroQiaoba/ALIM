import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from pll_model import PiCO
from models.cifar_resnet import CIFAR_ResNet
from models.cifar_preactresnet import CIFAR_PreActResNet
from utils.utils_algo import *
from utils.utils_loss import partial_loss, SupConLoss
from datasets.cifar10 import load_cifar10
from datasets.cifar100 import load_cifar100
from datasets.cifar100H import load_cifar100H
from datasets.cub200 import load_cub200
def train(args, epoch, train_loader,model, loss_fn, loss_cont_fn, optimizer):

    total_num = 0
    cls_bingo_num = 0
    cons_bingo_num = 0
    cont_labels_bingo_num = 0
    total_indexes = []
    total_plabels = []
    total_dlabels = []
    total_classfy_out = []
    total_cluster_out = []
    piror_set = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    cluster_piror_set_bingo_num = [0,0,0,0,0,0,0,0,0,0,0]
    classfy_piror_set_bingo_num = [0,0,0,0,0,0,0,0,0,0,0]
    
    model.train()
    margin = []
    clean_sample =[]
    for  i,((images_w1, images_s1, plabels, dlabels, index),(images_w2,_,_,_, index_2)) in enumerate(zip(train_loader,train_loader)):

        start_time = time.time()
        images_w1 = images_w1.cuda() # images_w1: [3, 32, 32]
        images_w2 = images_w2.cuda()
        images_s1 = images_s1.cuda()
        plabels = plabels.cuda()
        dlabels = dlabels.long().detach().cuda() # only for evalaution
        index = index.cuda()
        index_2 = index_2.cuda()
        
        # train and save results
        classfy_out, cluster_out, cont_features, cont_labels = model(images_w1, images_s1, plabels, args)
        total_num += plabels.size(0)
        cls_bingo_num  += torch.eq(torch.max(classfy_out, 1)[1], dlabels).sum().cpu()
        cons_bingo_num += torch.eq(torch.max(cluster_out, 1)[1], dlabels).sum().cpu()
        cont_labels_bingo_num += torch.eq(torch.max(cluster_out * (plabels +args.piror*(1-plabels)),1)[1], dlabels).sum().cpu()
        for jj in range(len(piror_set)):
            cluster_piror_set_bingo_num[jj] = cluster_piror_set_bingo_num[jj] + torch.eq(torch.max(cluster_out * (plabels +piror_set[jj]*(1-plabels)),1)[1], dlabels).sum().cpu()
            classfy_piror_set_bingo_num[jj] = classfy_piror_set_bingo_num[jj] + torch.eq(torch.max(classfy_out * (plabels +piror_set[jj]*(1-plabels)),1)[1], dlabels).sum().cpu()
        total_indexes.append(index.detach().cpu().numpy())
        total_plabels.append(plabels.detach().cpu().numpy())
        total_dlabels.append(dlabels.detach().cpu().numpy())
        total_classfy_out.append(classfy_out.detach().cpu().numpy())
        total_cluster_out.append(cluster_out.detach().cpu().numpy())

        # loss function
        batch_size = classfy_out.shape[0]
        cont_labels = cont_labels.contiguous().view(-1, 1)
        cont_mask = torch.eq(cont_labels[:batch_size], cont_labels.T).float().cuda() # mask for SupCon
        if epoch >= args.proto_start: # update confidence
            if args.augmentation_type =='case1':
                if args.proto_type=='cluster':  pred = cluster_out 
                if args.proto_type=='classify': pred = classfy_out
            loss_fn.confidence_update(args, pred, index, plabels)
        
        loss_cont = loss_cont_fn(features=cont_features, mask=cont_mask, batch_size=batch_size)
        loss_cls = loss_fn(args, classfy_out, index) # need preds
        if args.loss_weight_mixup !=0:
            lam = np.random.beta(args.alpha, args.alpha)
            lam = max(lam, 1-lam)
            pseudo_label_1 = loss_fn.confidence[index]
            pseudo_label_2 = loss_fn.confidence[index_2]
            X_w_mix = lam * images_w1  + (1 - lam) * images_w2      
            pseudo_label_mix = lam * pseudo_label_1 + (1 - lam) * pseudo_label_2
            logits_mix, _ ,_= model.encoder_q(X_w_mix)
            pred_mix = torch.softmax(logits_mix, dim=1)
            loss_mixup = - ((torch.log(pred_mix) * pseudo_label_mix).sum(dim=1)).mean()
            if args.loss_type == 'SCE':
                rceloss = - ((torch.log(pseudo_label_mix + 1e-7 ) * pred_mix).sum(dim=1)).mean()
                loss_mixup = args.sce_alpha*loss_mixup + args.sce_beta*rceloss
        
            loss = loss_cls + args.loss_weight * loss_cont + args.loss_weight_mixup*loss_mixup
        else:
            loss = loss_cls + args.loss_weight * loss_cont

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        end_time =time.time()
        per_sample_time = (end_time - start_time) / plabels.size(0)
        margin += ((torch.max(cluster_out*plabels, 1)[0])/(1e-9+torch.max(cluster_out*(1-plabels), 1)[0])).tolist()
        clean_sample+= (plabels*(torch.nn.functional.one_hot(dlabels,args.num_class))).sum(dim=1).cpu().tolist()

    epoch_cls_acc = cls_bingo_num/total_num
    epoch_cont_acc = cons_bingo_num/total_num
    epoch_cont_label_acc = cont_labels_bingo_num/total_num
    for jj in range(len(piror_set)):
        cluster_piror_set_bingo_num[jj]=cluster_piror_set_bingo_num[jj]/total_num
        classfy_piror_set_bingo_num[jj]=classfy_piror_set_bingo_num[jj]/total_num
    total_indexes = np.concatenate(total_indexes)
    total_plabels = np.concatenate(total_plabels)
    total_dlabels = np.concatenate(total_dlabels)
    total_classfy_out = np.concatenate(total_classfy_out)
    total_cluster_out = np.concatenate(total_cluster_out)

    print (f'Epoch={epoch}/{args.epochs} Train classification acc={epoch_cls_acc:.4f} contrastive acc={epoch_cont_acc:.4f} plabel_update_acc={epoch_cont_label_acc:.4f} piror={args.piror:4f}')
    print (piror_set)
    print (cluster_piror_set_bingo_num)
    print (classfy_piror_set_bingo_num)
    train_save = {
        #'epoch_partial_rate':   epoch_partial_rate,
        #'epoch_bingo_rate':     epoch_bingo_rate,
        'epoch_cls_acc':        epoch_cls_acc,
        'epoch_cont_acc':       epoch_cont_acc,
        'total_indexes':        total_indexes,
        'total_plabels':        total_plabels,
        'total_dlabels':        total_dlabels,
        'total_classfy_out':    total_classfy_out,
        'total_cluster_out':    total_cluster_out,
    }

    if epoch>=args.piror_start:
        if args.noise_rate>0:
            if args.piror_auto == 'case1': #Adaptive adjustment 1
                args.piror = sorted(margin)[int(len(margin)*args.noise_rate)]
            else:  #linear function, setting piror_add=1,piror_max=t,can get fix piror
                args.piror = min(args.piror+args.piror_add,args.piror_max)
        else:
            args.piror = 0
    if args.max1 ==True: #Limit the model to self-training at most
        args.piror = min (args.piror,1)
    if epoch%100==1:   #for plot Distribution of the value in Eq. 6 for clean and noise subsets with increasing training iterations. We conduct experiments on CIFAR-10 (q = 0.3, η = 0.3) with e0 = 80.
        di = {}
        di['margin']=margin
        di['clean'] = clean_sample
        di['lambda']=args.piror
        np.save(str(epoch)+'epoch.npy', di)
    return train_save


def test(args, epoch, test_loader, model):
    test_preds = []
    test_labels = []
    test_probs = []
    test_hidden1 = []
    test_hidden2 = []
    with torch.no_grad():     
        model.eval()
        bingo_num = 0
        total_num = 0
        # images: [3, 32, 32]
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.cuda(), labels.cuda()
            logit, _, hidden1, hidden2 = model(images, eval_only=True)
            _, predicts = torch.max(logit, 1)
            total_num += images.size(0)
            bingo_num += torch.eq(predicts, labels).sum().cpu()
            test_preds.append(predicts.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            test_probs.append(logit.cpu().numpy())
            test_hidden1.append(hidden1.cpu().numpy())
            test_hidden2.append(hidden2.cpu().numpy())
        test_acc = bingo_num / total_num
        print(f'Epoch={epoch}/{args.epochs} Test accuracy={test_acc:.4f}, bingo_num={bingo_num},  total_num={total_num}')
        test_hidden1 = np.concatenate(test_hidden1)
        test_hidden2 = np.concatenate(test_hidden2)
        test_probs = np.concatenate(test_probs)
        test_preds = np.concatenate(test_preds)
        test_labels = np.concatenate(test_labels)
        test_save = {
            'test_hidden1': test_hidden1,
            'test_hidden2': test_hidden2,
            'test_probs': test_probs,
            'test_preds': test_preds,
            'test_labels': test_labels
        }
    return test_acc, test_save
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of noise partial label learning')

    ## input parameters
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name (cifar10)')
    parser.add_argument('--dataset_root', default='./dataset/CIFAR-10', type=str, help='download dataset')
    parser.add_argument('--partial_rate', default=0.0, type=float, help='ambiguity level (q)')
    parser.add_argument('--noise_rate', default=0.0, type=float, help='noise level (gt may not in partial set)')
    parser.add_argument('--augment_type', default='pico', type=str, help='augment_type')
    parser.add_argument('--noisy_type', default='flip', type=str, help='flip or pico')
    parser.add_argument('--workers', default=6, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--noise_rate_estimate', default=-1, type=float, help='noise level estimate,default as noise rate')
    ## model parameters
    parser.add_argument('--encoder', default='resnet', type=str, help='encoder: preact or resnet')
    parser.add_argument('--low_dim', default=128, type=int, help='embedding dimension')
    parser.add_argument('--num_class', default=10, type=int, help='number of class')
    parser.add_argument('--moco_m', default=0.999, type=float, help='momentum for updating momentum encoder')
    parser.add_argument('--moco_queue', default=8192, type=int, help='queue size; number of negative samples')
    parser.add_argument('--loss_weight', default=0.5, type=float, help='contrastive loss weight')
    parser.add_argument('--conf_ema_range', default='0.95,0.8', type=str, help='pseudo target updating coefficient (phi)')
    parser.add_argument('--vMF', default=0.07, type=float) # Using vmf distribution to transform clustering probability into classification probability
    # -------------- for confidence update --------------
    parser.add_argument('--proto_m', default=0.99, type=float, help='momentum for computing the momving average of prototypes')
    parser.add_argument('--proto_start', default=1, type=int, help = 'Start Prototype Updating')
    parser.add_argument('--proto_type', default='cluster', type=str, help = 'Correct type: cluster or classify')
    parser.add_argument('--proto_case', default='Case1', type=str, help = 'Correct case: Case1(onehot update) or Case2(prob update)')
    
    #-------------- for DALI --------------
    parser.add_argument('--piror', default=0, type=float, help = 'for DALI')
    parser.add_argument('--piror_start', default=0, type=float, help = 'for initial')
    parser.add_argument('--piror_auto', default='case1', type=str, help = 'There are three methods to adjust the piror, the first two are adaptive adjustment, and the third is linear function')
    parser.add_argument('--mepoch', default=0.1, type=float, help = 'for case 2')
    parser.add_argument('--piror_add', default=1, type=float, help = 'for case 3')
    parser.add_argument('--piror_max', default=1, type=float, help = 'for case 3')
    parser.add_argument('--max1',  action='store_true', default=False)
    
    # -------------- for loss function --------------
    parser.add_argument('--loss_type', default='CE', type=str, help='loss type in training: CE, CC, EXP, LWC, MAE, MSE, SCE, GCE')
    parser.add_argument('--lwc_weight', default=1.0, type=float, help='weight in lwc loss, choose from [1,2,3]')
    parser.add_argument('--sce_alpha', default=0.1, type=float, help='alpha in rec loss, choose from [0.01, 0.1, 1, 6]')
    parser.add_argument('--sce_beta', default=1.0, type=float, help='beta in rec loss, choose from [0.1, 1.0]')
    parser.add_argument('--gce_q', default=1.0, type=float, help='q in gce loss, choose from [0.1, 0.2, 0.3, ..., 1.0]')
    parser.add_argument('--alpha', default=4, type=float, help='for mixup')
    parser.add_argument('--loss_weight_mixup', default=1, type=float, help='for mixup')
    parser.add_argument('--augmentation_type', default='case1', type=str, help='augmemtation type from case1~case6, default=case3')

    ## optimizer parameters
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lr_adjust', default='Case1', type=str, help='Learning rate adjust manner: Case1 or Case2.')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='weight decay (default: 1e-5).')
    parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')

    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer for training: adam or sgd')
    parser.add_argument('--seed', help='seed', type=int, default=0)
    parser.add_argument('--savewhole', action='store_true', default=False, help='whether to save whole results')
    parser.add_argument('--save_root', help='where to save results', default='./savemodels', type=str, required=False)

    args = parser.parse_args()


    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]
  
    print(args)
    cudnn.benchmark = True  #set seed
    torch.set_printoptions(precision=2, sci_mode=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    print (f'====== Step1: Reading Data =======')
    if args.dataset == 'cifar10':     #load data
        args.num_class = 10
        train_loader,train_givenY, test_loader = load_cifar10(args)
    elif args.dataset == 'cifar100':
        args.num_class = 100
        train_loader,train_givenY, test_loader = load_cifar100(args)
    elif args.dataset == 'cifar100H':
        args.num_class = 100
        train_loader,train_givenY, test_loader = load_cifar100H(args)
    elif args.dataset == 'cub200':
        args.num_class = 200
        train_loader,train_givenY, test_loader = load_cub200(args)

    print (f'training samples: {len(train_loader.dataset)}')
    print (f'testing samples: {len(test_loader.dataset)}')

    # normalize train_givenY
    train_givenY = torch.FloatTensor(train_givenY)
    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
    confidence = train_givenY.float()/tempY
    confidence = confidence.cuda()
    print (confidence)
    if args.noise_rate_estimate!=-1:
        args.noise_rate = args.noise_rate_estimate
    # set loss functions
    loss_fn = partial_loss(confidence)
    loss_cont_fn = SupConLoss()
    

    print (f'====== Step2: Gaining model and optimizer =======')
    if args.encoder == 'resnet':
        model = PiCO(args, CIFAR_ResNet, pretrained=False) # pretrain is not suitable for cifar dataset
        if args.dataset == 'cub200':
            model = PiCO(args, CIFAR_ResNet, pretrained=True)
    elif args.encoder == 'preact':
        model = PiCO(args, CIFAR_PreActResNet, pretrained=False)
    model = model.cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=0.9, 
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, 
                                     weight_decay=args.weight_decay)


    print (f'====== Step3: Training and Evaluation =======')
    test_accs = []
    all_labels = []
    
    argsepochs = args.epochs+1
    for epoch in range(1, argsepochs):
        if args.lr_adjust == 'Case1':
            adjust_learning_rate_V1(args, optimizer, epoch)
        elif args.lr_adjust == 'Case2':
            adjust_learning_rate_V2(args, optimizer, epoch)
        train_save = train(args, epoch, train_loader,model, loss_fn, loss_cont_fn, optimizer)
        loss_fn.set_conf_ema_m(epoch, args)
        test_acc, test_save = test(args, epoch, test_loader, model)
        test_accs.append(test_acc)

    print (f'====== Step4: Saving =======')
    save_root = args.save_root
    if not os.path.exists(save_root): os.makedirs(save_root)

    ## gain suffix_name
    modelname = 'pico'
    suffix_name = f'{args.dataset}_modelname:{modelname}_plrate:{args.partial_rate}_noiserate:{args.noise_rate}_loss:{args.loss_type}_model:{args.encoder}'
    ## gain res_name
    best_index = np.argmax(np.array(test_accs))
    bestacc = test_accs[best_index]
    res_name = f'testacc:{bestacc}'

    save_path = f'{save_root}/{suffix_name}_{res_name}_{time.time()}.npz'
    print (f'save results in {save_path}')
    np.savez_compressed(save_path,
                        args=np.array(args, dtype=object),
                        all_labels=all_labels,
                        )

