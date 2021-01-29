import os
import logging
import argparse
from datetime import datetime
from uuid import uuid4
from PIL import Image
from os import listdir
import numpy as np
from torch.utils.data import DataLoader
from torchsummary import summary
from PIL import Image
import torch
from torch.utils.data import Dataset
from models import generator, discriminator
from quatmodels import generator as quat_generator
from quatmodels import discriminator as quat_discriminator
from datetime import datetime

class BessarionMini(Dataset):
    def __init__(self, 
            base_folder: str,
            partition_name: str,
            subsetname: str = '',
            file_extension: str = 'resampled1024',
            virtual_partition: bool = False, # If True, training and test partitioning is determined in this initializer.
                                             # If False, the user must predetermine the partition manually by defining corresponding training/test folders
        ):
        def count_total_images(folder, file_extension):
            total_images = 0
            for fn in listdir(folder):
                if(file_extension not in fn):
                    continue
                total_images += 1
            return(total_images)

        if partition_name not in ['train', 'test']: raise ValueError
        self.data_filename_list = []
        if virtual_partition:
            inputpath = '{}/{}'.format(base_folder, subsetname)
        else:
            inputpath = '{}/{}/{}'.format(base_folder, partition_name, subsetname)
        gtpath = '{}/{}'.format(inputpath, 'text_nontext')
        total_images = count_total_images(inputpath, file_extension)
        print('Counted a total of *{}* images available.'.format(total_images))

        idx = -1 #(enumerate won't work here, because it takes into account all files regardless of extension)
        cutoff_pct = 0.8
        for fn in listdir(inputpath):
            #if(fn[-len(file_extension):] != file_extension):
            #    continue
            if(file_extension not in fn):
                continue
            idx += 1
            if(virtual_partition):
                if(partition_name == 'train' and idx > np.round( cutoff_pct * total_images) ): #TODO: This has to change to sth more generic
                    continue # 54 is approx. 80% of 67 images in bessarion-midi
                elif(partition_name == 'test' and idx <= np.round( cutoff_pct * total_images) ):
                    continue
                else:
                    pass
            print('Adding {} to *{}* fold.'.format(fn, partition_name))
            new_image = '{}/{}'.format(inputpath, fn)
            fn_gt = os.path.splitext(fn)[0] + '.npz'
            new_gt = '{}/{}'.format(gtpath, fn_gt)
            self.data_filename_list.append(
                (new_image, new_gt)
            )

    def __getitem__(self, index):
        mydatum = self.data_filename_list[index]
        datum_image = np.array(Image.open(mydatum[0]), dtype=np.float32) / 255.0
        datum_image = datum_image[:, :, 0:3]
        if(os.path.splitext(mydatum[1])[-1] == '.npz'):
            tt = np.load(mydatum[1])
            datum_gt = np.float32(tt['annotation_matrix'])
        else: #treat it as a uint8 image
            datum_gt = np.array(Image.open(mydatum[1]), dtype=np.float32) / 255.0
        # datum_image = distort_image_using_some_transform(datum_image) #Augmentation
        # Postprocess for Pytorch..
        datum_gt = datum_gt[:, :, None]
        datum_image = np.transpose(datum_image, [2, 0, 1])
        datum_gt = np.transpose(datum_gt, [2, 0, 1])
        return(datum_image, datum_gt)

    def __len__(self):
        return(len(self.data_filename_list))


def unique_id():
    """
    unique_id
        This is used to create a unique string to be used as a folder name and model instances.
    """
    return(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-') + uuid4().hex[0:5])

def learning_rate_step_parser(lrs_string):
    return [(int(elem.split(':')[0]), float(elem.split(':')[1])) for elem in lrs_string.split(',')]

def augment_to_4_channels(x, device):
    N, _, H, W = x.shape
    paddingchannel = torch.zeros(N, 1, H, W).to(device)
    x = torch.cat([x, paddingchannel], 1)        
    return(x)

def intersection_over_union(A, B):
    def binarize(x):
        x[x >= .5] = 1.
        x[x <  .5] = 0.
        return(x == 1.)
    Anp = binarize(A.detach().cpu().numpy())
    Bnp = binarize(B.detach().cpu().numpy())
    # Binarize inputs (the gt should be binarized anyway)
    intersection = np.logical_and(Anp, Bnp)
    union = np.logical_or(Anp, Bnp)
    res = np.count_nonzero(intersection) / np.count_nonzero(union)
    return(res)

def train():
    logger = logging.getLogger('QuaternionGAN::train')
    logger.info('========= Quaternionic GAN train ==========')
    logger.info('========= G.Sfikas  October 2020 ==========')
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                        help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
    parser.add_argument('--lr_gen', '-lrg', type=learning_rate_step_parser, default='100:1e-4,200:1e-6',
                        help='')
    parser.add_argument('--lr_dis', '-lrd', type=learning_rate_step_parser, default='100:5e-4,200:1e-5',
                        help='')
    parser.add_argument('--momentum', '-mom', action='store', type=float, default=0.5, #was:0.9
                        help='The momentum for SGD training (or beta1 for Adam). Default: 0.5')
    parser.add_argument('--momentum2', '-mom2', action='store', type=float, default=0.999,
                        help='Beta2 if solver is Adam. Default: 0.999')
    parser.add_argument('--weight_decay', '-wd', action='store', type=float, default=0.00005,
                        help='The weight decay for SGD training. Default: 0.00005')
    parser.add_argument('--lam', type=float, default=1000.0, help='L1 loss weighting parameter')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max number of epochs')    
    parser.add_argument('--ganversion', '-g', required=False, choices=['QGAN', 'VGAN'], default='QGAN',
                        help='NN version. Choose between Quaternionic GAN vs Vanilla GAN (QGAN,VGAN). Default: QGAN.')
    parser.add_argument('--gpu_id', '-gpu', action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='0',
                        help='The ID of the GPU to use. Default: GPU 0.')
    parser.add_argument('--base_num_channels', type=int, default=64, help='Base number of channels')    
    parser.add_argument('--debug', dest='debug', action='store_true', help='Debug mode.')
    parser.add_argument('--print_network', dest='print_network', action='store_true', help='Just print network info.')
    parser.set_defaults(debug=False,
        print_network=False,
    )
    args = parser.parse_args()
    logger.info('###########################################')
    for key, value in vars(args).items():
        logger.info('%s: %s', str(key), str(value))
    logger.info('###########################################')
    if not torch.cuda.is_available() or args.gpu_id[0] == -1:
        args.gpu_id = None
        device = torch.device('cpu')
        logger.warning('Could not find CUDA environment, using CPU mode (device: {})'.format(device))
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id[0]))
        logger.warning('Using GPU mode. Specifically our device is {}'.format(device))

    #############################################
    # Initialize training/test partitions
    #############################################
    trainingset = BessarionMini(base_folder = 'fixtures/bessarion-midi', partition_name = 'train', virtual_partition=True)
    testset = BessarionMini(base_folder = 'fixtures/bessarion-midi', partition_name = 'test', virtual_partition=True)
    #trainingset = BessarionMini(base_folder = 'bessarion-mini2', partition_name = 'train', subsetname= 'mini2', virtual_partition=False, file_extension='resampled100.png')
    #testset = BessarionMini(base_folder = 'bessarion-mini2', partition_name = 'test', subsetname= 'mini2', virtual_partition=False, file_extension='resampled100.png')
    logger.info('Training set now contains {} images.'.format(len(trainingset)))
    logger.info('Test set now contains {} images.'.format(len(testset)))

    train_loader = DataLoader(trainingset, shuffle=True, batch_size=1, num_workers=2)
    test_loader = DataLoader(testset, shuffle=False, batch_size=1, num_workers=1) #no shuffle in order to be able to compare loss/evaluation history

    #############################################
    # Create network
    #############################################    
    if(args.ganversion == 'VGAN'):
        G = generator(args.base_num_channels)
        D = discriminator(args.base_num_channels)
    elif(args.ganversion == 'QGAN'):
        G = quat_generator(args.base_num_channels)
        D = quat_discriminator(args.base_num_channels)


    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G = G.to(device)
    D = D.to(device)
    G.train()
    D.train()
    if(args.print_network):
        logger.info('Printing info for a hypothetical 1024x768 input to the network.')
        summary(G, (4, 1024, 768))
        summary(D, [(3, 1024, 768), (1, 1024, 768)])
        generator_total_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
        discriminator_total_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
        print('Generator total trainable params = {}, Discriminator total trainable params = {}'.format(
            generator_total_params,
            discriminator_total_params,
        ))
        print('Total trainable params = {}'.format(
            generator_total_params + discriminator_total_params
        ))
        exit(0)


    BCE_loss = torch.nn.BCELoss(size_average=True).to(device)
    L1_loss = torch.nn.L1Loss(size_average=True).to(device)
    BCE_loss_test = torch.nn.BCELoss(size_average=True).to(device) #not sure if I can just use the above one here

    ## Create optimizers and scheduler
    if args.solver_type == 'SGD':
        G_optimizer = torch.optim.SGD(G.parameters(), args.lr_gen[0][1],
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        D_optimizer = torch.optim.SGD(D.parameters(), args.lr_dis[0][1],
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.solver_type == 'Adam':
        G_optimizer = torch.optim.Adam(G.parameters(), args.lr_gen[0][1],
                                    weight_decay=args.weight_decay,
                                    betas=(args.momentum, args.momentum2))
        D_optimizer = torch.optim.Adam(D.parameters(), args.lr_dis[0][1],
                                    weight_decay=args.weight_decay,
                                    betas=(args.momentum, args.momentum2))
    else:
        raise NotImplementedError

    G_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, 'min', patience=2, verbose=True)
    D_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(D_optimizer, 'min', patience=2, verbose=True)

    ##########################
    # Init other params
    ##########################
    evaluation_iou_history = []
    evaluation_bceloss_history = []
    d_loss_history = []
    g_loss_history = []
    executionid = '{}-lam{}-glr{}-dlr{}-{}'.format(
        args.ganversion,
        args.lam,
        args.lr_gen[0][1],
        args.lr_dis[0][1],
        unique_id(),
    )
    logger.info('Using execution id: {} (search the subfolder of the same name in the results repertoire)'.format(
        executionid,
    ))
    os.mkdir('results/{}'.format(executionid))
    for epoch in range(1, args.max_epochs+1):
        ##############################################################################
        # Finetune trainable params in each epoch
        ##############################################################################
        for input_image, input_labels in train_loader:
            #print('Training input in iteration {}: {},{}'.format(epoch, input_image.shape, input_labels.shape))
            #Load to GPU
            input_image, input_labels = input_image.to(device), input_labels.to(device)
            input_image_augmented = augment_to_4_channels(input_image, device) #Necessary for quaternionic GAN

            # train discriminator D
            D.zero_grad()

            D_result = D(input_image, input_labels).squeeze()
            D_real_loss = BCE_loss(D_result, torch.ones(D_result.size()).to(device)  )

            G_result = G(input_image_augmented)
            D_result = D(input_image, G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, torch.zeros(D_result.size()).to(device)   )

            D_train_loss = D_real_loss + D_fake_loss # * 0.5
            D_train_loss.backward()
            D_optimizer.step()

            # train generator G
            G.zero_grad()
            G_result = G(input_image_augmented)
            D_result = D(input_image, G_result).squeeze()

            G_train_loss_adversarialpart = BCE_loss(D_result, torch.ones(D_result.size()).to(device) )
            G_train_loss_l1part = L1_loss(G_result, input_labels)
            G_train_loss = G_train_loss_adversarialpart + args.lam * G_train_loss_l1part
            G_train_loss.backward()
            G_optimizer.step()

            print('Epoch: {:03d} -----  D_train_loss: {:.3f}  G_train_loss: {:.3f} ({:.3f} + {:.3f}*{:.3f})'.format(
                epoch, 
                D_train_loss, 
                G_train_loss,
                G_train_loss_adversarialpart,
                args.lam,
                G_train_loss_l1part)
            )
            d_loss_history.append(D_train_loss.detach().cpu().numpy())
            g_loss_history.append(G_train_loss.detach().cpu().numpy())
        ##############################################################################
        # Check binary cross-entropy for the test images at the end of the epoch.
        ##############################################################################
        G.eval()
        evaluation_bceloss = []
        evaluation_iou = []
        evaluation_time = []
        for idx, (input_image, input_labels) in enumerate(test_loader):
            input_image, input_labels = input_image.to(device), input_labels.to(device)
            input_image_augmented = augment_to_4_channels(input_image, device)
            time_startevaluation = datetime.now()
            estimate_map = G(input_image_augmented)
            time_endevaluation = datetime.now()
            evaluation_time.append(time_endevaluation - time_startevaluation)
            estimate_map_numpy = estimate_map.squeeze().detach().cpu().numpy()
            Image.fromarray(np.uint8(estimate_map_numpy * 255.)).save(
                'results/{}/byzantine_epoch{:02d}_id{:02d}.png'.format(executionid, epoch, idx)
            )
            A = torch.flatten(input_labels)
            B = torch.flatten(estimate_map.detach())
            evaluation_bceloss.append(BCE_loss_test(A, B).cpu().numpy() )
            evaluation_iou.append(intersection_over_union(A, B) )
        # END FOR
        print(evaluation_time)
        logger.info('Evaluation took {} seconds.'.format(
            np.mean(evaluation_time)
        ))
        print('Mean test loss (bce on ground truth): {}'.format(np.mean(evaluation_bceloss)))
        print('Mean test IoU (bce on ground truth): {}'.format(np.mean(evaluation_iou)))
        evaluation_bceloss_history.append(evaluation_bceloss)
        evaluation_iou_history.append(evaluation_iou)
        G.train()
        ## Schedulers..
        G_scheduler.step(torch.from_numpy(np.mean(evaluation_bceloss, keepdims=True)))
        D_scheduler.step(torch.from_numpy(np.mean(evaluation_bceloss, keepdims=True)))
    # END FOR EPOCH
    np.savez('results/{}/results.npz'.format(executionid),
        bceloss = evaluation_bceloss_history,
        iou = evaluation_iou_history,
        d_loss = d_loss_history,
        g_loss = g_loss_history,
    )



if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    train()
