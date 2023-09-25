import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from utils.randaugment import RandomAugment

def load_cifar100H(args):
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    temp_train = dsets.CIFAR100(root=args.dataset_root, train=True, download=True)
    data, labels = temp_train.data, torch.Tensor(temp_train.targets).long()


    test_dataset = dsets.CIFAR100(root=args.dataset_root, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False)   

    partialY = generate_hierarchical_cv_candidate_labels(args, 'cifar100', labels, args.partial_rate, noisy_rate=args.noise_rate)

    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('Running defualt PLL setting')
    else:
        print('Running noisy PLL setting')
    print('Average candidate num: ', partialY.sum(1).mean())
    partial_matrix_dataset = Augmentention(data, partialY.float(), labels.float(), train_flag=True,args=args)

    
    partial_matrix_train_loader = torch.utils.data.DataLoader(
        dataset=partial_matrix_dataset, 
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True, 
        drop_last=True)
    return partial_matrix_train_loader,partialY,test_loader

class Augmentention(Dataset):
    def __init__(self, images, plabels, dlabels, train_flag=True,args=None):
        self.images = images
        self.plabels = plabels
        self.dlabels = dlabels
        self.train_flag = train_flag
        if args!=None:
            self.augment_type = args.augment_type
        normalize_mean = (0.5071, 0.4865, 0.4409)
        normalize_std  = (0.2673, 0.2564, 0.2762)
        if self.train_flag == True:
            self.weak_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(), 
                transforms.Normalize(normalize_mean, normalize_std) # the mean and std on cifar training set
                ])
            self.strong_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                RandomAugment(3, 5),
                transforms.ToTensor(), 
                transforms.Normalize(normalize_mean, normalize_std) # the mean and std on cifar training set
                ])
        else:
            self.test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
                ])


    def __len__(self):
        return len(self.dlabels)
        
    def __getitem__(self, index):
        if self.train_flag == True:
            each_image_w1 = self.weak_transform(self.images[index])
            each_image_s1 = self.strong_transform(self.images[index])
            each_plabel = self.plabels[index]
            each_dlabel = self.dlabels[index]
            return each_image_w1, each_image_s1, each_plabel, each_dlabel, index
        else:
            each_image = self.test_transform(self.images[index])
            each_dlabel = self.dlabels[index]
            return each_image, each_dlabel

def generate_hierarchical_cv_candidate_labels(args, dataname, train_labels, partial_rate=0.1, noisy_rate=0):
    assert dataname == 'cifar100'

    meta = unpickle(args.dataset_root + '/cifar-100-python/meta')

    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
    label2idx = {fine_label_names[i]:i for i in range(100)}

    x = '''aquatic mammals#beaver, dolphin, otter, seal, whale
fish#aquarium fish, flatfish, ray, shark, trout
flowers#orchid, poppy, rose, sunflower, tulip
food containers#bottle, bowl, can, cup, plate
fruit and vegetables#apple, mushroom, orange, pear, sweet pepper
household electrical devices#clock, keyboard, lamp, telephone, television
household furniture#bed, chair, couch, table, wardrobe
insects#bee, beetle, butterfly, caterpillar, cockroach
large carnivores#bear, leopard, lion, tiger, wolf
large man-made outdoor things#bridge, castle, house, road, skyscraper
large natural outdoor scenes#cloud, forest, mountain, plain, sea
large omnivores and herbivores#camel, cattle, chimpanzee, elephant, kangaroo
medium-sized mammals#fox, porcupine, possum, raccoon, skunk
non-insect invertebrates#crab, lobster, snail, spider, worm
people#baby, boy, girl, man, woman
reptiles#crocodile, dinosaur, lizard, snake, turtle
small mammals#hamster, mouse, rabbit, shrew, squirrel
trees#maple_tree, oak_tree, palm_tree, pine_tree, willow_tree
vehicles 1#bicycle, bus, motorcycle, pickup truck, train
vehicles 2#lawn_mower, rocket, streetcar, tank, tractor'''

    x_split = x.split('\n')
    hierarchical = {}
    reverse_hierarchical = {}
    hierarchical_idx = [None] * 20
    # superclass to find other sub classes
    reverse_hierarchical_idx = [None] * 100
    # class to superclass
    super_classes = []
    labels_by_h = []
    for i in range(len(x_split)):
        s_split = x_split[i].split('#')
        super_classes.append(s_split[0])
        hierarchical[s_split[0]] = s_split[1].split(', ')
        for lb in s_split[1].split(', '):
            reverse_hierarchical[lb.replace(' ', '_')] = s_split[0]
            
        labels_by_h += s_split[1].split(', ')
        hierarchical_idx[i] = [label2idx[lb.replace(' ', '_')] for lb in s_split[1].split(', ')]
        for idx in hierarchical_idx[i]:
            reverse_hierarchical_idx[idx] = i

    # end generate hierarchical
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    transition_matrix = np.eye(K) * (1 - noisy_rate)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate
    mask = np.zeros_like(transition_matrix)
    for i in range(len(transition_matrix)):
        superclass = reverse_hierarchical_idx[i]
        subclasses = hierarchical_idx[superclass]
        mask[i, subclasses] = 1

    transition_matrix *= mask
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        random_n_j = random_n[j]
        while partialY[j].sum() == 0:
            random_n_j = np.random.uniform(0, 1, size=(1, K))
            partialY[j] = torch.from_numpy((random_n_j <= transition_matrix[train_labels[j]]) * 1)
    
    print("Finish Generating Candidate Label Sets!\n")
    return partialY

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res