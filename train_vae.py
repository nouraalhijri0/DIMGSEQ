"""
Train a diffusion model on images.
"""

import argparse
import json, torch, os
from torch import nn
import numpy as np
from diffuseq.utils import dist_util, logger
from diffuseq.text_datasets import load_data_text
from diffuseq.step_sample import create_named_schedule_sampler
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_model_emb,
    load_tokenizer
)
from train_util import TrainLoop
from transformers import set_seed
import wandb
from diffuseq.VAE_model import VAE, vae_loss
import timeit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from diffuseq.utils import logger

### custom your wandb setting here ###
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["WANDB_API_KEY"] = "705af41c4fd178ea1e7091a0b6ebd1eb82a5ff12"
os.environ["WANDB_MODE"] = "offline"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
output_dir = 'vae-models'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_loss_dict(desc, loss):
    logger.logkv_mean(desc, loss)
            
def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults) # update latest args according to argparse
    return parser

def train_one_epoch(epoch_index, model, training_loader, batch_size, lr, optimizer):
    loss_avg_list = np.array([])
    last_loss = 0.
    start = timeit.default_timer()
    
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    logger.log('### start of training one epoch')
    for i, data in enumerate(training_loader):
        logger.log(f'training batch {i} out of {len(training_loader)}')
        # Every data instance is an input + label pair
        _, data = data
        input_ids = data['input_ids'].float().cuda()
        input_ids = input_ids.unsqueeze(2)
        input_ids /= 59000

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        _, recons_input_ids = model(input_ids)

        # Compute the loss and its gradients
        loss = vae_loss(input_ids, recons_input_ids, model.module.Encoder)
        loss.backward()
        log_loss_dict('training batch loss: ', loss)
        logger.log(f'training batch {i} out of {len(training_loader)} -> Loss={loss}')
        loss_avg_list = np.append(loss_avg_list, loss.cpu().detach().numpy())
        running_loss = loss_avg_list.mean()
        # Adjust learning weights
        optimizer.step()
        #tqdm_object.set_postfix(train_loss=running_loss, lr=lr)
        
        # Gather data and report
    stop = timeit.default_timer()
    t = stop - start
    logger.log(f'### end of training one epoch, epoch loss={running_loss}, time={t}')
    return loss_avg_list, running_loss, t 

def val_one_epoch(epoch_index, model, training_loader, batch_size, lr):
    loss_avg_list = np.array([])
    last_loss = 0.
    start = timeit.default_timer()
    
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    logger.log('### start of validating one epoch')
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        _, data = data
        input_ids = data['input_ids'].float().cuda()
        input_ids = input_ids.unsqueeze(2)
        input_ids /= 59000

        # Make predictions for this batch
        _, recons_input_ids = model(input_ids)

        # Compute the loss and its gradients
        loss = vae_loss(input_ids, recons_input_ids, model.module.Encoder)
        log_loss_dict('validation batch loss: ', loss)
        logger.log(f'validating batch {i} out of {len(training_loader)} -> Loss={loss}')
        loss_avg_list = np.append(loss_avg_list, loss.cpu().detach().numpy())
        running_loss = loss_avg_list.mean()
        #tqdm_object.set_postfix(valid_loss=running_loss, lr=lr)
        
        # Gather data and report
    stop = timeit.default_timer()
    t = stop - start
    logger.log(f'### end of validating one epoch, epoch loss={running_loss}, time={t}')
    return loss_avg_list, running_loss, t 

def trainLoop(model, data, batch_size, lr, eval_data, args):
    
    EPOCH=2
    best_loss = float('inf')
    iterations = list(range(batch_size*EPOCH))
    all_train_loss, all_val_loss = np.array([]), np.array([])
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.001)

    for i in range(EPOCH):
        #training epoch
        model.train()
        loss_list_t, train_epoch_loss, t = train_one_epoch(i, model,data,batch_size,lr, optimizer)
        all_train_loss = np.append(all_train_loss, loss_list_t)
        
        #validation epoch
        model.eval()
        with torch.no_grad():
            loss_list_v, val_epoch_loss, t = val_one_epoch(i, model,eval_data,batch_size,lr)
            all_val_loss = np.append(all_val_loss, loss_list_v)
            
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            torch.save(model.module.state_dict(), f"{args.checkpoint_path}/VAE_model.pt")
            print("Saved Best Model!")
            
    all_train_loss, all_val_loss = all_train_loss.flatten(), all_val_loss.flatten()
    plt.title('Loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.plot(iterations, all_train_loss, color='blue')
    plt.plot(iterations, all_val_loss, color='green')
    plt.legend(['Training Loss','Validation Loss'])
    plt.savefig(f'{output_dir}/train_val_loss.png')
    plt.show()

    

def main():
    args = create_argparser().parse_args()
    set_seed(102) 
    dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    tokenizer = load_tokenizer(args)
    model_weight, tokenizer = load_model_emb(args, tokenizer)

    data = load_data_text(
        batch_size=2048,
        seq_len=args.seq_len,
        data_args = args,
        loaded_vocab=tokenizer,
        model_emb=model_weight # use model's weights as init
    )
    #####################

    data_valid = load_data_text(
        batch_size=2048,
        seq_len=args.seq_len,
        data_args=args,
        split='valid',
        deterministic=True,
        loaded_vocab=tokenizer,
        model_emb=model_weight # using the same embedding wight with tranining data
    )

    print('#'*30, 'size of vocab', args.vocab_size)

    logger.log("### Creating model and diffusion...")
    # print('#'*30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    model = VAE(input_dim=64, hidden_dim=768)
    
    
    # print('#'*30, 'cuda', dist_util.dev())
    gpu_id = range(torch.cuda.device_count())
    print(f'Number of devices is: {gpu_id}')
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[g for g in gpu_id]) #  DEBUG **
    # model.cuda() #  DEBUG **
    '''model = DDP(
                model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False#, static_graph=True
            )'''

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "ArtELingo_diffusion"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("### Training...")
    trainLoop(
        model=model,
        data=data,
        batch_size=128,
        lr=0.0001,
        eval_data=data_valid,
        args=args
    )

if __name__ == "__main__":
    main()


