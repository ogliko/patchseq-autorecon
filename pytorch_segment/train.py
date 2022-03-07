from neurotorch.datasets.specification import JsonSpec
from neurotorch.core.trainer import Trainer
from neurotorch.nets.RSUNet import RSUNet
import os
import numpy as np
import torch
import torch.optim as optim
import argparse

def train(ckpt, ckpt_dir, log_dir, json_dir, eps, epochs, num_stacks):
    inputs_list = [f for f in os.listdir(json_dir) if 'inputs' in f]
    inputs_list.sort()
    labels_list = [f for f in os.listdir(json_dir) if 'labels' in f]
    labels_list.sort()    

    # Initialize network and json specification
    net = RSUNet()
    json_spec = JsonSpec()

    # Define experiment name from arguments
    exp_name = str(json_dir.split('data/')[1]) + '_' + str(eps) + '_' + str(ckpt_dir)

    # Define checkpoints directory
    ckpt_dir = os.path.join('checkpoints', exp_name)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    # Define log directory
    log_dir = os.path.join(log_dir, exp_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    for k in range(epochs):
        # Create random indices
        idx = np.random.permutation(len(inputs_list))
        print(idx)
        
        for i in range(int(np.floor(len(idx)/num_stacks))):
            i1 = i*num_stacks
            i2 = i*num_stacks + num_stacks
            select = idx[i1:i2]
            print(i, i1, i2, select)
            spec1 = [json_spec.parse(os.path.join(json_dir, inputs_list[j]))[0] for j in select]
            spec2 = [json_spec.parse(os.path.join(json_dir, labels_list[j]))[0] for j in select]
                        
            validation_split = 0.01
            inputs_vol = []
            labels_vol = []
            inputs_vol_val = []
            labels_vol_val = []
            
            for s in range(len(spec1)):
                volume1 = []
                volume2 = []
                spec = [spec1[s]]
                inputs = json_spec.create(spec,stack_size=33)
                spec = [spec2[s]]
                labels = json_spec.create(spec,stack_size=33)
                
                for n in range(len(inputs)):
                    if not (labels[n].getArray() == 0).all():
                        volume1.append(inputs[n].getArray().astype(np.uint8))
                        volume2.append(labels[n].getArray().astype(np.uint8))
                
                del inputs
                del labels
                
                valid_indexes = np.arange(len(volume1))
                np.random.seed(0)
                random_idx = np.random.permutation(valid_indexes)
                val_idx = random_idx[int(len(valid_indexes)*(1-validation_split)):].copy()
                volume1_val = [volume1[ind] for ind in val_idx] # Create validation inputs volume
                volume2_val = [volume2[ind] for ind in val_idx] # Create validation labels volume
                
                for ind in sorted(val_idx, reverse=True): # Remove validation data from training volume
                    del volume1[ind]
                    del volume2[ind]
                
                inputs_vol = inputs_vol + volume1
                labels_vol = labels_vol + volume2
                inputs_vol_val = inputs_vol_val + volume1_val
                labels_vol_val = labels_vol_val + volume2_val
            
            inputs_vol = [inputs_vol, inputs_vol_val]
            labels_vol = [labels_vol, labels_vol_val]
            del inputs_vol_val
            del labels_vol_val

            # Initialize optimizer with updated epsilon parameter
            optimizer = optim.Adam(net.parameters(), eps=eps)

            # Initialize trainer
            if (i==0) & (ckpt == 'None'):
                trainer = Trainer(net, inputs_vol, labels_vol, checkpoint_dir=ckpt_dir, checkpoint_period=10000, 
                                     logger_dir=log_dir, max_epochs=1, gpu_device=0, optimizer=optimizer)       
            else:
                trainer = Trainer(net, inputs_vol, labels_vol, checkpoint_dir=ckpt_dir, checkpoint_period=10000, 
                                     logger_dir=log_dir, checkpoint=ckpt, max_epochs=1, gpu_device=0, optimizer=optimizer)
            
            cell_str = str()
            for s1 in select:
                cell_str = cell_str + '_{:03d}'.format(s1)
            print(k, i, cell_str, ckpt)

            # Begin training
            trainer.run_training()
            
            # Save the last model       
            trainer.save_checkpoint('last{:03d}_{:03d}{}.ckpt'.format(k, i, cell_str))            

            # Save the best model
            os.rename(os.path.join(ckpt_dir, 'best.ckpt'), os.path.join(ckpt_dir, 'best{:03d}_{:03d}{}.ckpt'.format(k, i, cell_str)))

            # Set a new checkpoint
            ckpt = os.path.join(ckpt_dir, 'last{:03d}_{:03d}{}.ckpt'.format(k, i, cell_str)) # Use the last model as a new checkpoint

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', '-ck', type=str, help='path to checkpoint')
    parser.add_argument('--ckpt_dir', '-c', type=str, help='directory to save checkpoint')
    parser.add_argument('--log_dir', '-l', type=str, help='directory to save logs')
    parser.add_argument('--json_dir', '-j', type=str, help='directory of json files')
    parser.add_argument('--eps', '-e', type=float, default=1e-1)
    parser.add_argument('--epochs', '-ep', type=int, default=10)
    parser.add_argument('--num_stacks', '-n', type=int, default=3)
    args = parser.parse_args()
    train(args.ckpt, args.ckpt_dir, args.log_dir, args.json_dir, args.eps, args.epochs, args.num_stacks)
