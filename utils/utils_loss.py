import torch
import torch.nn.functional as F
import torch.nn as nn

class partial_loss(nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99):
        super().__init__()
        self.confidence = confidence
        self.conf_ema_m = conf_ema_m

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * epoch / args.epochs * (end - start) + start

    # classfy_out: preds
    def forward(self, args, classfy_out, index):
        if args.loss_type == 'CE': # same with rc and proden loss
            pred = classfy_out
            target = self.confidence[index, :] 
            average_loss = - ((torch.log(pred) * target).sum(dim=1)).mean()

        elif args.loss_type == 'CC': # same with LOG
            pred = classfy_out
            partialY = self.confidence[index, :].clone()
            partialY[partialY > 0]  = 1 # [0,1,1,0,1,0]
            average_loss = - torch.log((pred * partialY).sum(dim=1)).mean()

        elif args.loss_type == 'EXP':
            pred = classfy_out
            partialY = self.confidence[index, :].clone()
            partialY[partialY > 0]  = 1 # [0,1,1,0,1,0]
            average_loss = torch.exp(-(pred * partialY).sum(dim=1)).mean()

        elif args.loss_type == 'LWC':
            sm_outputs = classfy_out
            partialY = self.confidence[index, :].clone()
            partialY[partialY > 0]  = 1 # [0,1,1,0,1,0]

            ## (onezero, counter_onezero)
            ## onezero: partial label
            ## counter_onezero: non-partial label
            onezero = torch.zeros(sm_outputs.shape[0], sm_outputs.shape[1])
            onezero[partialY > 0] = 1
            counter_onezero = 1 - onezero
            onezero = onezero.cuda()
            counter_onezero = counter_onezero.cuda()

            ## for partial loss (min) + weight (use ce to calculate loss function)
            sig_loss1 = - torch.log(sm_outputs + 1e-8)
            l1 = self.confidence[index, :] * onezero * sig_loss1 
            average_loss1 = torch.sum(l1) / l1.size(0)

            ## for non-partial loss (max) + weight
            sig_loss2 = - torch.log(1 - sm_outputs + 1e-8)
            l2 = counter_onezero * sig_loss2 / args.num_class
            average_loss2 = torch.sum(l2) / l2.size(0)

            average_loss = average_loss1 + args.lwc_weight * average_loss2

        elif args.loss_type == 'MAE':
            pred = classfy_out
            target = self.confidence[index, :]
            loss_fn = torch.nn.L1Loss(reduction='none')
            average_loss = loss_fn(pred, target).sum(dim=1).mean()

        elif args.loss_type == 'MSE':
            pred = classfy_out
            target = self.confidence[index, :]
            loss_fn = torch.nn.MSELoss(reduction='none')
            average_loss = loss_fn(pred, target).sum(dim=1).mean()

        elif args.loss_type == 'SCE':
            pred = classfy_out
            target = self.confidence[index, :]
            pred = torch.clamp(pred, min=1e-7, max=1.0)
            target = torch.clamp(target, min=1e-4, max=1.0) # avoid nan
            celoss = - ((torch.log(pred) * target).sum(dim=1)).mean()
            rceloss = - ((torch.log(target) * pred).sum(dim=1)).mean()
            average_loss = args.sce_alpha*celoss + args.sce_beta*rceloss

        elif args.loss_type == 'GCE': # without Truncate
            pred = classfy_out
            target = self.confidence[index, :]
            gceloss = (1 - torch.pow(pred, args.gce_q)) / args.gce_q 
            average_loss = (gceloss * target).sum(dim=1).mean()

        return average_loss
    
    # cluster results update partial weights
    def confidence_update(self, args, cluster_out, index, plabels):
        with torch.no_grad():
            if args.proto_case == 'Case1': #Onehot+DALI

                _, prot_pred = (cluster_out * (plabels + args.piror*(1 -plabels))).max(dim=1)
                pseudo_label = F.one_hot(prot_pred, plabels.shape[1]).float().cuda().detach()
            elif args.proto_case == 'Case2': # without DALI
                pseudo_label = cluster_out * plabels
                pseudo_label = pseudo_label / pseudo_label.sum(dim=1).repeat(pseudo_label.size(1),1).transpose(0,1) 
                pseudo_label = pseudo_label.float().cuda().detach()
            elif args.proto_case == 'Case3':#RC+DALI
                pseudo_label = cluster_out * (plabels + args.piror*(1 -plabels))
                pseudo_label = pseudo_label / pseudo_label.sum(dim=1).repeat(pseudo_label.size(1),1).transpose(0,1) 
                pseudo_label = pseudo_label.float().cuda().detach()
            self.confidence[index, :] = self.conf_ema_m * self.confidence[index, :] + (1 - self.conf_ema_m) * pseudo_label


class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning: 
        https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    ## features: A(.)
    ## mask: index for P(.)
    def forward(self, features, mask=None, batch_size=-1):
        #  device = torch.device('cuda') if features.is_cuda else torch.device('cpu')

        # SupCon loss (Partial Label Mode)
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if mask is not None:
            mask = mask.float().detach().cuda() # [size of q, size of A(.)]
            # compute logits 计算所有 k * A() / t
            anchor_dot_contrast = torch.div(        # [size of q, size of A(.)]
                torch.matmul(features[:batch_size], features.T),
                self.temperature)

            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # [size of q, 1]
            logits = anchor_dot_contrast - logits_max.detach() # [size of q, size of A(.)]

            # mask-out self-contrast cases
            '''
            tensor([[0., 1., 1.,  ..., 1., 1., 1.],
                    [1., 0., 1.,  ..., 1., 1., 1.],
                    [1., 1., 0.,  ..., 1., 1., 1.],
                    ...,
                    [1., 1., 1.,  ..., 1., 1., 1.],
                    [1., 1., 1.,  ..., 1., 1., 1.],
                    [1., 1., 1.,  ..., 1., 1., 1.]], device='cuda:0')
            '''
            logits_mask = torch.scatter(
                torch.ones_like(mask), # all ones
                1,
                torch.arange(batch_size).view(-1, 1).cuda(),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size*2]
            queue = features[batch_size*2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators [postive 在index=0的位置]
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss