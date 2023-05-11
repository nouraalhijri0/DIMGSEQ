"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch as th
import torch
from torch import nn
import torch.distributed as dist
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round, get_weights
from diffuseq.text_datasets import load_data_text
from diffuseq.VAE_model import VAE
from diffuseq.step_sample import create_named_schedule_sampler


# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_model_emb,
    load_tokenizer
)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0)
    decode_defaults = dict(split='test', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()
    # load configurations.
    args.model_path = 'diffusion_models/diffuseq_qqp_h128_lr0.0001_t1000_sqrt_lossaware_seed102_test-artelingo20230501-21:33:54'
    config_path = os.path.join(args.model_path, "training_args.json")

    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    logger.log("### Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )
    
    model.load_state_dict(th.load(os.path.join(args.model_path, "ema_0.9999_010000.pt"), map_location=th.device('cpu')))

    lm_head = VAE(64, 768)
    lm_head.load_state_dict(torch.load('vae-checkpoint/VAE_model.pt'))
    model.lm_head = lm_head
    model.input_up_proj = nn.Identity()
    model.word_embedding = lm_head.Encoder
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model.eval()

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    tokenizer = load_tokenizer(args)

    model_emb, tokenizer = load_model_emb(args, tokenizer)

    #model_emb.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())
    model_emb_copy = model_emb.cpu()#get_weights(model_emb, args)

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    ## load data
    data_valid = load_data_text(
        batch_size=1,
        seq_len=args.seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=model_emb.cpu(), # using the same embedding wight with tranining data
        loop=False
    )
    
    start_t = time.time()
    
    # batch, cond = next(data_valid)
    # print(batch.shape)

    '''model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")'''
    # fout = open(out_path, 'a')

    all_test_data = []

    #all_test_data.append(args.sequence)
    try:
        for i in range(1):
            batch, cond = next(data_valid)
            all_test_data.append(cond)

    except StopIteration:
        print('### End of reading iteration...')
    '''all_test_data[0]['input_ids'] = all_test_data[0]['input_ids'][:100,:]
    all_test_data[0]['input_mask'] = all_test_data[0]['input_mask'][:100,:]
    all_test_data[0]['pixel_values'] = all_test_data[0]['pixel_values'][:100,:]
    all_test_data[0]['encoded_txt'] = all_test_data[0]['encoded_txt'][:100,:]
    
    all_test_data[1]['input_ids'] = all_test_data[1]['input_ids'][:100,:]
    all_test_data[1]['input_mask'] = all_test_data[1]['input_mask'][:100,:]
    all_test_data[1]['pixel_values'] = all_test_data[1]['pixel_values'][:100,:]
    all_test_data[1]['encoded_txt'] = all_test_data[1]['encoded_txt'][:100,:]'''

    from tqdm import tqdm
    principalComponents_x = []
    principalComponents_y = []
    
    for cond in tqdm(all_test_data):
        input_ids_x = cond.pop('input_ids').to(dist_util.dev())
        x_start = model.get_embeds(input_ids_x)
        input_ids_mask = cond.pop('input_mask')
        input_ids_mask_ori = input_ids_mask
        noise = th.randn_like(x_start)
        input_ids_mask = th.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(dist_util.dev())
        x_noised = th.where(input_ids_mask==0, x_start, noise)
        model_kwargs = cond

        if args.step == args.diffusion_steps:
            args.use_ddim = False
            step_gap = 1
        else:
            args.use_ddim = True
            step_gap = args.diffusion_steps//args.step


        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4,4))
  
        input_ids_x = torch.randint(0,60000,size=(128,)) / 60000
        t = torch.LongTensor(list(range(1000)))
        
        #pca = PCA(n_components=2)
        #principalComponents = pca.fit_transform(x_start.cpu().detach().numpy().squeeze())
        #x_start = torch.tensor(principalComponents).to(dist_util.dev())
        noise = torch.randn(size=input_ids_x.shape).to(input_ids_x.device)
        #######################################
        t = torch.LongTensor([0]).to(input_ids_x.device)
        a = torch.tensor([diffusion.sqrt_alphas_cumprod[t]]).to(input_ids_x.device)
        a_c = torch.tensor([diffusion.sqrt_one_minus_alphas_cumprod[t]]).to(input_ids_x.device)
        x_t = a * input_ids_x + a_c * noise
        x_t = x_t.cpu().detach().numpy().squeeze()
        plt.title('t=0')
        plt.xlabel('random tokens')
        plt.ylabel('tokens ids')
        plt.scatter(list(range(x_t.shape[0])), x_t, s=10, c='b')
        plt.savefig('diff_step_0.png')
        
        t = torch.LongTensor([499]).to(input_ids_x.device)
        a = torch.tensor([diffusion.sqrt_alphas_cumprod[t]]).to(input_ids_x.device)
        a_c = torch.tensor([diffusion.sqrt_one_minus_alphas_cumprod[t]]).to(input_ids_x.device)
        x_t = a * input_ids_x + a_c * noise
        x_t = x_t.cpu().detach().numpy().squeeze()
        plt.title('t=T/2')
        plt.xlabel('random tokens')
        plt.ylabel('tokens ids')
        plt.scatter(list(range(x_t.shape[0])), x_t, s=10, c='b')
        plt.savefig('diff_step_500.png')
        
        t = torch.LongTensor([999]).to(input_ids_x.device)
        a = torch.tensor([diffusion.sqrt_alphas_cumprod[t]]).to(input_ids_x.device)
        a_c = torch.tensor([diffusion.sqrt_one_minus_alphas_cumprod[t]]).to(input_ids_x.device)
        x_t = a * input_ids_x + a_c * noise
        x_t = x_t.cpu().detach().numpy().squeeze()
        plt.title('t=T')
        plt.xlabel('random tokens')
        plt.ylabel('tokens ids')
        plt.scatter(list(range(x_t.shape[0])), x_t, s=10, c='b')
        plt.savefig('diff_step_999.png')
        


if __name__ == "__main__":
    main()


