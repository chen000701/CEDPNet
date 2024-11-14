import torch
import time
from torch.optim import Adam
# import network
from loss import structure_loss
import numpy as np
import cv2
import logger
from numpy import mean
from torch import nn
from dataset import get_loader
from os.path import join
import random
import os
from utils import mkdir, write_doc, get_time
from utils_net import S_measure
from tensorboardX import SummaryWriter
from model.mynet import Network

# def resize(input, target_size=(224, 224)):
#     return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)

np.set_printoptions(suppress=True, threshold=1e5)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(7)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=50):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*decay
        print('decay_epoch: {}, Current_LR: {}'.format(decay_epoch, init_lr*decay))


class Solver(object):
    def __init__(self):
        self.ICNet = Network().cuda()

    def train(self, roots, val_roots, init_epoch, end_epoch, learning_rate, batch_size, val_batch_size, weight_decay, ckpt_root, doc_path, log_dir, num_thread, pin, vgg_path=None):
        # Define Adam optimizer.
        optimizer = Adam(self.ICNet.parameters(),
                         lr=learning_rate, 
                         weight_decay=weight_decay)
        
        # adjust_lr(optimizer, learning_rate, epoch, opt.decay_rate, opt.decay_epoch)
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logger_net = logger.create_logger(log_dir, name="ORSI-CoSOD")
        
        # Load ".pth" to initialize model.
        if init_epoch == 0:
            print('init_epoch == 0')
            # From pre-trained VGG16.
            # self.ICNet.apply(network.weights_init)
            # ICNet.load_pre('/root/hy-tmp/TCNet-main/TCNet-main/smt_tiny.pth')
            # print('load_state_dict!!!')
            # self.ICNet.vgg.vgg.load_state_dict(torch.load(vgg_path))
            # self.ICNet.rgb.load_state_dict(torch.load('/hy-tmp/TCNet-main/TCNet-main/80.7_T2T_ViT_t_14.pth.tar'))
        else:
            # From the existed checkpoint file.
            ckpt = torch.load(join(ckpt_root, 'MyNet_Weights_{}.pth'.format(init_epoch)))
            self.ICNet.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])

        # Define training dataloader.
        train_dataloader = get_loader(roots=roots,
                                      request=('img', 'gt'),
                                      shuffle=True,
                                      batch_size=batch_size,
                                      data_aug=True,
                                      num_thread=num_thread,
                                      pin=pin)
        
        val_loader = get_loader(roots=val_roots,
                              request=('img', 'gt'),
                              shuffle=False,
                              batch_size=val_batch_size,
                              data_aug=True,
                              num_thread=1,
                              pin=False)
        
        writer = SummaryWriter(os.path.join(ckpt_root, 'tensorboard'))
        # best_mae = 1
        best_epoch = 0
        best_Sm = 0
        
        # Train.
        self.ICNet.train()
        for epoch in range(init_epoch + 1, end_epoch + 1):
            print("epoch：%s / %s" % (epoch, end_epoch))
            
            adjust_lr(optimizer, learning_rate, epoch, 0.1, 50)
            
            start_time = get_time()
            loss_sum = 0.0

            for data_batch in train_dataloader:
                self.ICNet.zero_grad()

                # Obtain a batch of data.
                img, gt = data_batch['img'], data_batch['gt']
                img, gt = img.cuda(), gt.cuda()

                if len(img) == 1:
                    # Skip this iteration when training batchsize is 1 due to Batch Normalization. 
                    continue
                
                # Forward.
                S_1_pred, S_2_pred, S_3_pred, S_4_pred, S_5_pred = self.ICNet(img)

                # gt = nn.Sigmoid(gt)
                # Compute IoU loss.
                loss1 = structure_loss(S_1_pred, gt)
                loss2 = structure_loss(S_2_pred, gt)
                loss3 = structure_loss(S_3_pred, gt)
                loss4 = structure_loss(S_4_pred, gt)
                loss5 = structure_loss(S_5_pred, gt)
                loss = 5*loss1 + 4*loss2 + 3*loss3 + 2*loss4 + 1*loss5
                # Backward.
                loss.backward()
                optimizer.step()
                loss_sum = loss_sum + loss.detach().item()
            
            if not os.path.exists(ckpt_root):
                os.makedirs(ckpt_root)
            # Save the checkpoint file (".pth") after each epoch.
            # mkdir(ckpt_root)
            # torch.save({'optimizer': optimizer.state_dict(),
            #             'state_dict': self.ICNet.state_dict()}, join(ckpt_root, 'E_Weights_{}.pth'.format(epoch)))
            
            # Compute average loss over the training dataset approximately.
            loss_mean = loss_sum / len(train_dataloader)
            end_time = get_time()

            # Record training information (".txt").
            content = 'CkptIndex={}:    TrainLoss={}    LR={}    Time={}\n'.format(epoch, loss_mean, optimizer.param_groups[0]['lr'], end_time - start_time)
            write_doc(doc_path, content)
            
            if (epoch % 5 == 0) and (epoch > 10):
                print(f"===> Validation on epoch: {str(epoch)} <===")
                with torch.no_grad():
                    psnr_list = []
                    # edge_mae_list = []
                    recon_psnr_list = []
                    Sm_list = []
                    
                    self.ICNet.eval()
                    for data_batch in val_loader:
                        # self.ICNet.zero_grad()

                        # Obtain a batch of data.
                        image, binary = data_batch['img'], data_batch['gt']
                        image, binary = image.cuda(), binary.cuda()

                        S_1_pred, S_2_pred, S_3_pred, S_4_pred, S_5_pred = self.ICNet(image)
                        frame_out = S_1_pred
                        # print(frame_out.shape)
                        # print(binary.shape)
                        # frame_out = F.interpolate(frame_out, size=binary.shape[2:], mode='bilinear', align_corners=True)

                        # if not os.path.exists(output_dir):
                        #     os.makedirs(output_dir)
        #                 imwrite(frame_out, output_dir +'/' +str(batch_idx) + '.png', range=(0, 1))
        #                 mae = torch.abs(frame_out[0]-clean).mean()
        #                 edge_mae = torch.abs(frame_out[2]-edge).mean()
        #                 Sm = S_measure(frame_out[0], clean).cuda()
        #                 psnr_list.append(mae)
        #                 edge_mae_list.append(edge_mae)
        #                 Sm_list.append(Sm)

                        mae = torch.abs(frame_out - binary).mean()
                        #edge_mae = torch.abs(frame_out- edge).mean()
                        Sm = S_measure(frame_out, binary).cuda()
                        psnr_list.append(mae)
                        #edge_mae_list.append(edge_mae)
                        Sm_list.append(Sm)


                    avr_mae = sum(psnr_list) / len(psnr_list)
                    #avr_edge_mae = sum(edge_mae_list) / len(edge_mae_list)
                    avr_Sm = sum(Sm_list) / len(Sm_list)

                    # frame_debug = torch.cat((frame_out[0],clean), dim =0)
                    # edge_img = torch.cat((frame_out[2], edge), dim=0)
                    # m_s_map = torch.cat((frame_out[1], frame_out[0]), dim=0)

                    frame_debug = torch.cat((frame_out, binary), dim=0)
                    #edge_img = torch.cat((frame_out[2], edge), dim=0)
                    #m_s_map = torch.cat((frame_out[1], frame_out[0]), dim=0)

                    writer.add_images('salient_image_and_label', frame_debug, epoch)
                    # writer.add_images('edge_image_and_label', edge_img, epoch)
                    # writer.add_images('moderate_salient_map_and_pred_map', m_s_map, epoch)

                    writer.add_scalars('salient_mae_testing', {'test salient_mae':avr_mae.item(),
                                                           }, epoch)
                    # writer.add_scalars('edge_testing', {'test edge_mae': avr_edge_mae.item(),
                    #                                        }, epoch)
                    writer.add_scalars('salient_Sm_testing', {'test salient_Sm': avr_Sm.item(),
                                                           }, epoch)
                    #torch.save(MyEnsembleNet.state_dict(), os.path.join(args.model_save_dir,'epoch'+ str(epoch) + '.pkl'))
                    if best_Sm < avr_Sm:
                        best_Sm = avr_Sm
                        best_epoch = epoch
                        torch.save(self.ICNet.state_dict(), os.path.join(ckpt_root,'epoch'+ str(epoch) + '.pkl'))

                        logger_net.info(f"epoch: {epoch}\t"
                                f"salient_mae: {avr_mae.item():.6f}\t"
                                f"salient_Sm: {avr_Sm.item():.6f}\t"
                                f"best_epoch: {best_epoch}\t"
                                f"best_Sm: {best_Sm.item():.6f}\t"
                                f"lr: {optimizer.param_groups[0]['lr']}")
        writer.close()

            
    
    def test(self, roots, ckpt_path, pred_root, num_thread, batch_size, original_size, pin):
        with torch.no_grad():            
            # Load the specified checkpoint file(".pth").
            state_dict = torch.load(ckpt_path)
            self.ICNet.load_state_dict(state_dict)
            self.ICNet.eval()
            
            # Get names of the test datasets.
            datasets = roots.keys()

            # Test ICNet on each dataset.
            for dataset in datasets:
                # Define test dataloader for the current test dataset.
                test_dataloader = get_loader(roots=roots[dataset], 
                                             request=('img', 'file_name', 'group_name', 'size'),
                                             shuffle=False,
                                             data_aug=False, 
                                             num_thread=num_thread, 
                                             batch_size=batch_size, 
                                             pin=pin)

                # Create a folder for the current test dataset for saving predictions.
                mkdir(pred_root)
                cur_dataset_pred_root = join(pred_root, dataset)
                mkdir(cur_dataset_pred_root)

                for data_batch in test_dataloader:
                    # Obtain a batch of data.
                    img = data_batch['img'].cuda()

                    time_list = []
                    start_each = time.time()
                    # Forward.
                    S_1_pred, S_2_pred, S_3_pred, S_4_pred, S_5_pred = self.ICNet(img)
                    preds = S_1_pred
                    time_each = time.time() - start_each
                    
                    print(time_each)
                    #print("{}'s average Time Is : {:.1f} fps".format(1 / mean(time_list)))
                    
                    # Create a folder for the current batch according to its "group_name" for saving predictions.
                    group_name = data_batch['group_name'][0]
                    cur_group_pred_root = join(cur_dataset_pred_root, group_name)
                    mkdir(cur_group_pred_root)

                    # preds.shape: [N, 1, H, W]->[N, H, W, 1]
                    preds = preds.permute(0, 2, 3, 1).cpu().numpy()

                    # Make paths where predictions will be saved.
                    pred_paths = list(map(lambda file_name: join(cur_group_pred_root, file_name + '.png'), data_batch['file_name']))
                    
                    # For each prediction:
                    for i, pred_path in enumerate(pred_paths):
                        # Resize the prediction to the original size when "original_size == True".
                        H, W = data_batch['size'][0][i], data_batch['size'][1][i]
                        pred = cv2.resize(preds[i], (int(W), int(H))) if original_size else preds[i]

                        # Save the prediction.
                        cv2.imwrite(pred_path, np.array(pred * 255))
