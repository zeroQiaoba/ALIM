import time
import torch
import torch.nn as nn
import numpy as np
from random import sample
import torch.nn.functional as F


class PiCO(nn.Module):

    def __init__(self, args, base_encoder, pretrained=False):
        super().__init__()

        self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, pretrained=pretrained)
        self.encoder_k = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, pretrained=pretrained)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim))
        self.register_buffer("queue_pseudo", torch.randn(args.moco_queue))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))        
        self.register_buffer("prototypes", torch.zeros(args.num_class,args.low_dim))
        self.vMF  = args.vMF
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        batch_size = x.shape[0]
        idx_shuffle = torch.randperm(batch_size).cuda()
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        return x[idx_unshuffle]


    def forward(self, img_q, img_k=None, plabel=None, args=None, eval_only=False):
        
        ## logit: classification result (without plabel)
        classfy_out, q, hiddens = self.encoder_q(img_q)
        classfy_out = torch.softmax(classfy_out, dim=1)
        
        ## cluster_out: cluster result (without plabel)
        prototypes = self.prototypes.clone().detach()
        logits_prot = torch.mm(q, prototypes.t())
        cluster_out = torch.softmax(logits_prot/self.vMF, dim=1)

        if eval_only:
            return classfy_out, cluster_out, q, hiddens


        predicetd_scores =  classfy_out * (plabel +args.piror*(1-plabel)) 

        _, pseudo_labels = torch.max(predicetd_scores, dim=1)

        ## pseudo_labels -> cluster center
        for feat, label in zip(q, pseudo_labels):
            prototypes[label] = prototypes[label]*args.proto_m + (1-args.proto_m)*feat    
        self.prototypes = F.normalize(prototypes, dim=1) # normalize prototypes

        ## pseudo_labels -> contrastive learning
        with torch.no_grad():
            self._momentum_update_key_encoder(args) # update the momentum encoder
            img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)
            _, k, _ = self.encoder_k(img_k)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        cont_features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        cont_labels = torch.cat((pseudo_labels, pseudo_labels, self.queue_pseudo.clone().detach()), dim=0)
        self._dequeue_and_enqueue(k, pseudo_labels, args)

        # classfy_out: classification result (without plabel) => preds
        # cluster_out: clustering result     (without plabel) => preds
        # (cont_features, cont_labels): SupCon
        return classfy_out, cluster_out, cont_features, cont_labels

