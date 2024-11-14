import os
from solver import Solver

# vgg_path = '/root/TCNet-main/TCNet-main/vgg16_feat.pth'
ckpt_root = './ckpt_FINAL_net_v3/'
train_init_epoch = 0
train_end_epoch = 80
train_device = '0'
train_doc_path = './training_FINAL_net_v3.txt'
log_dir='./results/Log_FINAL_net_v3'
learning_rate = 1e-4
weight_decay = 1e-4
train_batch_size = 8
train_num_thread = 8

# An example to build "train_roots".
train_roots = {'img': '/root/CoORSI/train/images/',
               'gt': '/root/CoORSI/train/labels/'}
# ------------- end -------------

val_roots = {'img': '/root/CoORSI/test/images/',
             'gt': '/root/CoORSI/test/labels/'}
# ------------- end -------------

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = train_device
    solver = Solver()
    solver.train(roots=train_roots,
                 val_roots=val_roots,
                 # vgg_path=vgg_path,
                 init_epoch=train_init_epoch,
                 end_epoch=train_end_epoch,
                 learning_rate=learning_rate,
                 batch_size=train_batch_size,
                 val_batch_size=1,
                 weight_decay=weight_decay,
                 ckpt_root=ckpt_root,
                 doc_path=train_doc_path,
                 log_dir=log_dir,
                 num_thread=train_num_thread,
                 pin=False)


