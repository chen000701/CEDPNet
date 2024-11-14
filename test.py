import os
from solver_seed_loss_2 import Solver

test_device = '0'
test_batch_size = 8
pred_root = '/root/0_data_result/pred/final_v3-40/'
ckpt_path = '/root/MyNet/MyNet-main/ckpt_FINAL_net_v3/epoch40.pkl'
original_size = True
test_num_thread = 8

# An example to build "test_roots".
test_roots = dict()
datasets = ['CoORSI']

for dataset in datasets:
    roots = {'img': '/root/{}/test/images/'.format(dataset)}
    test_roots[dataset] = roots
# ------------- end -------------

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = test_device
    solver = Solver()
    solver.test(roots=test_roots,
                ckpt_path=ckpt_path,
                pred_root=pred_root, 
                num_thread=test_num_thread, 
                batch_size=test_batch_size, 
                original_size=original_size,
                pin=False)
