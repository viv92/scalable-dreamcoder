import os
import math 
from matplotlib import pyplot as plt 
import matplotlib.colors as mcolors
import numpy as np
import torch
torch.set_float32_matmul_precision('high') # use TF32 precision for speeding up matmul
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json 
from time import time 
import random 

from transformers import RobertaTokenizer, T5ForConditionalGeneration
from utils_scdr_slicedLatents_taskEmbFrozen_noFantasy import *
from python_sandbox import *


def reconstruction_loss(recon_x, x):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    recon_x = recon_x.permute(0,2,1) # [b, vocab, seqlen]
    recon_loss = criterion(recon_x, x)
    return recon_loss

def compression_loss(z):
    criterion = nn.MSELoss(reduction='mean')
    # latent compress loss - drive latents to zero (pad) latent
    compress_targets = torch.zeros_like(z)
    compress_loss = criterion(z, compress_targets)
    return compress_loss

def task_reconstruction_loss(recon_x, x):
    criterion = nn.MSELoss(reduction='mean')
    recon_loss = criterion(recon_x, x)
    return recon_loss

def ema(arr, val, r=0.01):
    if len(arr) == 0:
        return [val]
    newval = arr[-1] * (1-r) + val * r 
    arr.append(newval)
    return arr 

# utility function to freeze model
def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()


# utility function to load model weights from checkpoint - loads to the device passed as 'device' argument
def load_ckpt(checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device('cpu'), mode='eval'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if mode == 'eval':
        model.eval() 
        return model
    else:
        model.train()
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            return model, optimizer, scheduler
        else:
            return model, optimizer
        
# utility function to save a checkpoint (model_state, optimizer_state, scheduler_state) - saves on cpu (to save gpu memory)
def save_ckpt(device, checkpoint_path, model, optimizer, scheduler=None):
    # transfer model to cpu
    model = model.to('cpu')
    # prepare dicts for saving
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, checkpoint_path)
    # load model back on original device 
    model = model.to(device)

# utility function to check if string is a number
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False



## main 
if __name__ == '__main__':

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # init program description / caption embedder model (codeT5)
    t5_tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    t5_model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base').to(device)
    t5_d_model = 768
    t5_vocab_size = t5_tokenizer.vocab_size # 32100
    t5_max_seq_len = t5_tokenizer.model_max_length # 512 # TODO this might require shortening the program descriptions
    # delete t5_decoder to save ram 
    del t5_model.decoder 
    # freeze t5_encoder
    freeze(t5_model.encoder)

    # hyperparams for vocab and bottleneck dims 
    dit_vocab_size = t5_vocab_size + 1
    dit_mask_token = dit_vocab_size - 1

    # hyperparams for latent slicing 
    program_latent_seqlen = t5_max_seq_len

    # hyperparams for dit 
    dit_d_model = 512 
    dit_n_heads = 8
    assert dit_d_model % dit_n_heads == 0
    dit_d_k = dit_d_model // dit_n_heads
    dit_d_v = dit_d_k 
    dit_n_layers = 6 
    dit_d_ff = dit_d_model * 4

    dropout = 0.1 
    weight_decay = 0.01 # 0.1 

    # hyperparams for training 
    diffusion_start_time_eps = 1e-3
    batch_size = 32
    gradient_accumulation_steps = 8 
    lr = 1e-4 # 3e-4
    num_train_steps = 10 ** 7
    train_steps_done = 0
    random_seed = 10
    resume_training_from_ckpt = False               

    # hyperparams for figures and plotting
    plot_freq = 10 ** 4
    sampling_freq = int(plot_freq / 4)

    # hyperparams for sampling
    num_sampling_steps = int(program_latent_seqlen * 0.25)
    p_uncond = 0.1 
    cfg_scale = 2.0
    sampling_eps = 0.1 # 0.3 # 0

    # create hyperparam str
    hyperparam_dict = {}
    hyperparam_dict['method'] = 'scdr_onlyDiT' 
    hyperparam_dict['pLatSeq'] = program_latent_seqlen
    hyperparam_dict['B'] = batch_size 
    hyperparam_dict['lr'] = lr
    hyperparam_dict['Wdecay'] = weight_decay
    # hyperparam_dict['drop'] = dropout
    hyperparam_dict['ditD'] = dit_d_model
    # hyperparam_dict['RF'] = reconstruction_factor
    # hyperparam_dict['WF'] = wake_factor
    # hyperparam_dict['initMode'] = sleep_mode
    # hyperparam_dict['sleep'] = sleep_steps
    # hyperparam_dict['sampSteps'] = num_sampling_steps
    # hyperparam_dict['delay'] = delay_start_iter 
    # hyperparam_dict['Fnum'] = max_fantasies 
    # hyperparam_dict['Fskips'] = fantasy_gen_skip_cycles 
    # hyperparam_dict['timeout'] = timeout 

    hyperparam_str = ''
    for k,v in hyperparam_dict.items():
        hyperparam_str += '|' + k + '=' + str(v)

    results_dir = './results/' + hyperparam_str + '/'
    ckpts_dir = './ckpts/'
    dit_ckpt_path = ckpts_dir + 'dit_' + hyperparam_str + '.pt'
    train_buffer_ckpt_path = ckpts_dir + 'trainBuf_' + hyperparam_str + '.pt'
      
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)

    # set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # load data
    folder = './'
    # train_dataset_path = folder + 'filtered-dataset/filtered-dataset-0.pkl'
    # test_dataset_path = folder + 'filtered-dataset/filtered-dataset-1.pkl'
    train_dataset_path = folder + 'filtered-dataset/filtered-hq-deduped-python.pkl'

    with open(train_dataset_path, 'rb') as f:
        train_dataset = pickle.load(f)

    # with open(test_dataset_path, 'rb') as f:
    #     test_dataset = pickle.load(f)

    # de-duplicate dataset based on problem statements
    seen_prompts = set() 
    deduped_dataset = []
    for x in train_dataset:
        if x['prompt'] not in seen_prompts:
            deduped_dataset.append(x)
            seen_prompts.add(x['prompt'])
    train_dataset = deduped_dataset
    print('len(deduped_dataset): ', len(train_dataset))

#     # convert all prompts to fantasy prompts (only inputs and outputs)
#     for i, data in enumerate(train_dataset):
#         fantasy_prompt = '''Provide a PYTHON3 solution to the following problem:

# You are given some input and output pairs as examples below. Infer the common pattern or rule that maps all the inputs to their respective outputs. 

# The format of input and output has to be inferred from the examples as well.'''

#         fantasy_prompt += '\n\nExamples\n\n'
#         for j, x in enumerate(data['inputs']):
#             x = x.strip()
#             fantasy_prompt += 'Input\n\n' + x + '\n\n'
#             fantasy_prompt += 'Output\n\n' + data['outputs'][j] + '\n\n'
#         train_dataset[i]['prompt'] = fantasy_prompt 


    # start with a small train dataset 
    np.random.shuffle(train_dataset)

    # init dit 
    dit_x_seq_len = program_latent_seqlen
    dit_max_seq_len = dit_x_seq_len + 1 # [t, x]
    dit_condition_dim = t5_d_model
    dit = init_dit(dit_max_seq_len, dit_x_seq_len, dit_d_model, dit_condition_dim, dit_vocab_size, dit_d_k, dit_d_v, dit_n_heads, dit_n_layers, dit_d_ff, dropout, device).to(device)

    # init optimizers
    dream_params = dit.parameters()
    dream_optimizer = torch.optim.AdamW(dream_params, lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)

    # init sandbox 
    sandbox = Sandbox(default_timeout=2)

    # load dit if resume 
    if resume_training_from_ckpt:
        dit, dream_optimizer = load_ckpt(dit_ckpt_path, dit, optimizer=dream_optimizer, device=device, mode='train')

    # results for plotting
    results_dit_loss, results_dit_success_accuracy, results_dit_correct_accuracy, results_fantasy_fraction, results_fantasy_correct_fraction, results_wake_dit_loss, results_edit_fantasy_fraction = [], [], [], [], [], [], []
    results_train_loss, results_recon_loss, results_compress_loss = [], [], []
    results_codebook_usage, results_codebook_unique, results_masked_latents = [], [], []
    results_infuse_wake, results_infuse_fant = [], []

    # train
    train_step = train_steps_done
    pbar = tqdm(total=num_train_steps)

    dit.train()

    # start DiT training
    while train_step < num_train_steps + train_steps_done:

        dream_on_what = -1
        minibatch = []

        if len(train_dataset) > 0:
            # fetch train minibatch
            idx = np.arange(len(train_dataset))
            np.random.shuffle(idx)
            idx = idx[:batch_size]
            minibatch += [train_dataset[i] for i in idx]

        if len(minibatch) > 0:
            dream_on_what = 1

            if dream_on_what != -1:

                # prepare program tokens
                with torch.no_grad():
                    program_batch = [item['code'] for item in minibatch]
                    program_tokens_dict = t5_tokenizer(program_batch, return_tensors='pt', padding='max_length', truncation=True)
                    # TODO use obtained attention mask for fsq encoder 
                    program_tokens, program_attn_mask = program_tokens_dict.input_ids, program_tokens_dict.attention_mask

                    # get caption embeddings
                    cap_batch = [item['prompt'] for item in minibatch]
                    cap_tokens_dict = t5_tokenizer(cap_batch, return_tensors='pt', padding=True, truncation=True)
                    cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
                    cap_embs = t5_model.encoder(input_ids=cap_tokens.to(device), attention_mask=cap_attn_mask.to(device)).last_hidden_state 

                ## loss for DiT 

                x = program_tokens.to(device) # x.shape: [b, seq_len] 
                condition = cap_embs

                # set condition = None with prob p_uncond
                if np.random.rand() < p_uncond: # TODO: explore the effect of no CFG versus CFG only during training versus CFG during training and sampling
                    condition = None

                # sample diffusion time ~ uniform(eps, 1)
                t = (1 - diffusion_start_time_eps) * torch.rand(x.shape[0], device=device) + diffusion_start_time_eps

                # get noise from noise schedule
                sigma, dsigma = logLinearNoise(t)

                # perturb the data
                x_perturb = perturb(x, sigma, dit_mask_token)

                # use bfloat16 precision for speed up # NOTE RoPE gives wrong results with bfloat16
                # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

                # get score
                log_score = dit(x_perturb, sigma, condition)

                # calculate loss 
                dit_loss = score_entropy_loss(log_score, sigma.unsqueeze(-1), x_perturb, x, dit_mask_token)
                dit_loss = (dsigma.unsqueeze(-1) * dit_loss).sum(dim=-1).mean()

                ## total dream loss 
                dream_loss = dit_loss 

                # adjustment for gradient accumulation 
                loss_scaled = dream_loss / gradient_accumulation_steps
                loss_scaled.backward()

                if (train_step + 1) % gradient_accumulation_steps == 0:
                    # gradient cliping - helps to prevent unnecessary divergence 
                    torch.nn.utils.clip_grad_norm_(dream_params, max_norm=1.0)
                    # gradient step
                    dream_optimizer.step()
                    dream_optimizer.zero_grad()

                # bookeep losses
                results_dit_loss = ema(results_dit_loss, dream_loss.item()) 

                if len(results_dit_loss) > 0:
                    pbar.set_description('dit_loss: {:.3f}'.format(results_dit_loss[-1]))

        pbar.update(1)


        # # sampling
        # if (train_step+1) % sampling_freq == 0:

        #     # put model in eval mode to avoid dropout
        #     dit.eval()

        #     # generate sample from dit
        #     if dream_on_what != -1:

        #         with torch.no_grad():

        #             # get caption embeddings
        #             cap_batch = [item['prompt'] for item in minibatch]
        #             cap_tokens_dict = t5_tokenizer(cap_batch, return_tensors='pt', padding=True, truncation=True)
        #             cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
        #             cap_embs = t5_model.encoder(input_ids=cap_tokens.to(device), attention_mask=cap_attn_mask.to(device)).last_hidden_state 

        #             # get sample tokens corresponding to indices of codebook 
        #             pred_program_tokens = get_sample(dit, program_latent_seqlen, dit_mask_token, dit_vocab_size, num_sampling_steps, cap_embs.shape[0], cap_embs, cfg_scale, 0, device) # x_sample.shape: [b, seqlen]

        #             # decode programs
        #             pred_programs = t5_tokenizer.batch_decode(pred_program_tokens, skip_special_tokens=True)

        #             # execute programs 
        #             success_runs = 0
        #             correct_runs = 0
        #             for i in range(len(minibatch)):
        #                 data = minibatch[i]
        #                 program = pred_programs[i]
        #                 success = 1
        #                 correct = 1
        #                 for j, x in enumerate(data['inputs']):
        #                     # format input correctly
        #                     x = x.strip()
        #                     x = x.splitlines()
        #                     # execute
        #                     output, error = sandbox.execute(program, inputs=x)
        #                     if error:
        #                         success *= 0
        #                         break 
        #                     else: # check for correctness 
        #                         gt_output = data['outputs'][j].strip()
        #                         if (output is not None) and (output.strip() != ''):
        #                             output = output.strip()
        #                         eq = output == gt_output
        #                         correct *= eq 
        #                 if (len(data['inputs']) > 0) and success:
        #                     success_runs += 1
        #                     if correct:
        #                         correct_runs += 1

        #             # get solved minibatch idx
        #             success_accuracy = success_runs / len(minibatch)
        #             correct_accuracy = correct_runs / len(minibatch)
        #             results_dit_success_accuracy.append(success_accuracy)
        #             results_dit_correct_accuracy.append(correct_accuracy)

        #     dit.train()


        ## plotting
        if (train_step+0) % plot_freq == 0: ## save ckpt and plot losses

            # save dit ckpt 
            save_ckpt(device, dit_ckpt_path, dit, dream_optimizer)

            # plot dit loss
            if len(results_dit_loss) > 0:

                fig = plt.figure()
                plt.plot(results_dit_loss, label='dit_loss')
                plt.legend()
                plt.title('final_loss:{:.3f}'.format(results_dit_loss[-1]))
                plt.ylim([0, 1000])
                save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(2) + '_ditLoss.png'
                fig.savefig(save_path)
                plt.close(fig)

            # plot dit batch_accuracy
            if len(results_dit_success_accuracy) > 0:

                fig = plt.figure()
                plt.plot(results_dit_success_accuracy, label='dit_success_accuracy')
                plt.plot(results_dit_correct_accuracy, label='dit_correct_accuracy')
                plt.legend()
                plt.title('success:{:.3f} correct:{:.3f}'.format(results_dit_success_accuracy[-1], results_dit_correct_accuracy[-1]))
                save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(2) + '_ditAccuracy.png'
                fig.savefig(save_path)
                plt.close(fig)

            # plot fantasy fraction
            if len(results_fantasy_fraction) > 0:

                fig = plt.figure()
                plt.plot(results_fantasy_fraction, label='fantasy_fraction')
                plt.plot(results_fantasy_correct_fraction, label='fantasy_correct_fraction')
                plt.plot(results_edit_fantasy_fraction, label='edit_fantasy_fraction')
                plt.plot(results_infuse_fant, label='infusion')
                plt.legend()
                plt.title('fant:{:.3f} correct:{:.3f} edit:{:.3f}'.format(results_fantasy_fraction[-1], results_fantasy_correct_fraction[-1], results_edit_fantasy_fraction[-1]))
                save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(2) + '_fantFrac.png'
                fig.savefig(save_path)
                plt.close(fig)


        train_step += 1

    pbar.close()