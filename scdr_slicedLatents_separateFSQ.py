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

    # hyperparams for quantization
    num_quantized_values = [7, 7, 7, 5, 5, 5] # L in fsq paper (use odd values only)
    latent_dim = len(num_quantized_values)

    # hyperparams for latent slicing 
    program_latent_seqlen = 128 # 32

    # hyperparams for fsq 
    fsq_d_model = 512 
    fsq_n_heads = 8
    assert fsq_d_model % fsq_n_heads == 0
    fsq_d_k = fsq_d_model // fsq_n_heads 
    fsq_d_v = fsq_d_k 
    fsq_n_layers = 6 # 3
    fsq_d_ff = fsq_d_model * 4

    dropout = 0.1 
    weight_decay = 0.01 # 0.1 
    compress_factor = 1 # 0.01 # 0
    reconstruction_factor = 1
    wake_factor = 0 # 1 

    # hyperparams for training 
    batch_size = 16 # 8
    gradient_accumulation_steps = 8 # NOTE try increasing this
    lr = 3e-4
    num_train_steps = 10 ** 7
    train_steps_done = 0
    random_seed = 10
    resume_training_from_ckpt = False             

    # hyperparams for figures and plotting
    plot_freq = 10 ** 4

    # hyperparams for fantasies
    num_edit_fantasies = 10 ** 1 # 3
    all_edit_fantasy_buffer_size = 10 ** 3 # 2
    train_buffer_size = 10 ** 5 # 3

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # create hyperparam str
    hyperparam_dict = {}
    hyperparam_dict['method'] = 'scdr_separateFSQ_noEditFant'  # 'scdr_separateFSQ_editFantOneTrainDataset' 
    hyperparam_dict['latentD'] = latent_dim
    hyperparam_dict['fsqD'] = fsq_d_model
    hyperparam_dict['fsqL'] = fsq_n_layers
    hyperparam_dict['pLatSeq'] = program_latent_seqlen
    hyperparam_dict['B'] = batch_size 
    hyperparam_dict['gradAcc'] = gradient_accumulation_steps
    hyperparam_dict['lr'] = lr
    hyperparam_dict['Wdecay'] = weight_decay
    # hyperparam_dict['drop'] = dropout
    hyperparam_dict['CF'] = compress_factor
    # hyperparam_dict['RF'] = reconstruction_factor
    # hyperparam_dict['WF'] = wake_factor
    # hyperparam_dict['initMode'] = sleep_mode
    # hyperparam_dict['sleep'] = sleep_steps
    # hyperparam_dict['sampSteps'] = num_sampling_steps
    # hyperparam_dict['delay'] = delay_start_iter 
    # hyperparam_dict['Fnum'] = max_fantasies 
    # hyperparam_dict['Fskips'] = fantasy_gen_skip_cycles 
    # hyperparam_dict['editFbuf'] = all_edit_fantasy_buffer_size 
    hyperparam_dict['trainbuf'] = train_buffer_size 
    # hyperparam_dict['timeout'] = timeout 

    hyperparam_str = ''
    for k,v in hyperparam_dict.items():
        hyperparam_str += '|' + k + '=' + str(v)


    results_dir = './results/' + hyperparam_str + '/'
    ckpts_dir = './ckpts/'
    dit_ckpt_path = ckpts_dir + 'dit_' + hyperparam_str + '.pt'
    fsq_ckpt_path = ckpts_dir + 'fsq_' + hyperparam_str + '.pt'
    train_buffer_ckpt_path = ckpts_dir + 'train_buf_' + hyperparam_str + '.pt'
      
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

    # init FSQ 
    fsq_encoder = init_fsq_encoder_transformer(t5_vocab_size, t5_d_model, t5_max_seq_len, fsq_d_model, fsq_d_k, fsq_d_v, fsq_n_heads, fsq_n_layers, fsq_d_ff, dropout, latent_dim, device)
    fsq_decoder = init_fsq_decoder_transformer(latent_dim, t5_max_seq_len, fsq_d_model, fsq_d_k, fsq_d_v, fsq_n_heads, fsq_n_layers, fsq_d_ff, dropout, t5_vocab_size, device)
    fsq = FSQ_Transformer(device, num_quantized_values, fsq_encoder, fsq_decoder, t5_max_seq_len, program_latent_seqlen).to(device)

    # init optimizers
    sleep_params = fsq.parameters()
    sleep_optimizer = torch.optim.AdamW(sleep_params, lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)

    # init sandbox 
    sandbox = Sandbox(default_timeout=2)

    # load ckpt
    if resume_training_from_ckpt:
        fsq, sleep_optimizer = load_ckpt(fsq_ckpt_path, fsq, sleep_optimizer, device=device, mode='train')
        with open(train_buffer_ckpt_path, 'rb') as f:
            train_dataset = pickle.load(f)

    # results for plotting
    results_dit_loss, results_dit_success_accuracy, results_dit_correct_accuracy, results_fantasy_fraction, results_fantasy_correct_fraction, results_wake_dit_loss, results_edit_fantasy_fraction = [], [], [], [], [], [], []
    results_train_loss, results_recon_loss, results_compress_loss = [], [], []
    results_codebook_usage, results_codebook_unique, results_masked_latents = [], [], []
    results_infuse_wake, results_infuse_fant = [], []

    # train
    train_step = train_steps_done
    pbar = tqdm(total=num_train_steps)

    new_train_dataset = []
    # new_train_dataset = train_dataset
    # train_dataset = []

    all_edit_fantasy_dataset = [] 
    edit_fantasy_dataset = []
    edit_fantasy_visualize_text = ''


    # generate fantasies by editing train programs
    fsq.eval()
    with torch.no_grad():

        edit_fantasy_dataset = []
        pbar2 = tqdm(total=num_edit_fantasies)
        fant_start_time = time()

        while len(edit_fantasy_dataset) < num_edit_fantasies:

            pbar2.set_description('Generating edit fantasies')

            # fetch train minibatch
            train_idx = np.arange(len(train_dataset))
            np.random.shuffle(train_idx)
            train_idx = train_idx[:batch_size]
            minibatch = [train_dataset[i] for i in train_idx]

            for data in minibatch:

                edited_code = mutate_program(data['code'])
                if edited_code != 0:
                    success = 1
                    correct = 1
                    gen_outputs = []
                    for j, x in enumerate(data['inputs']):
                        # format input correctly
                        x = x.strip()
                        x = x.splitlines()
                        # execute
                        output, error = sandbox.execute(edited_code, inputs=x)
                        if error:
                            success *= 0
                            break 
                        else: # check for correctness 
                            gt_output = data['outputs'][j].strip()
                            if (output is not None) and (output.strip() != ''):
                                output = output.strip()
                                gen_outputs.append(output)
                            eq = output == gt_output
                            correct *= eq 
                    if (len(data['inputs']) > 0) and success and (not correct):
                        if (len(gen_outputs) == len(data['inputs'])):
                            # create fantasy 
                            fantasy_prompt = '''Provide a PYTHON3 solution to the following problem:

You are given some input and output pairs as examples below. Infer the common pattern or rule that maps all the inputs to their respective outputs. 

The format of input and output has to be inferred from the examples as well.'''

                            fantasy_prompt += '\n\nExamples\n\n'
                            num_fantasy_examples = 0 
                            for j, x in enumerate(data['inputs']):

                                # skip if output is very long string (else tokenization becomes very slow)
                                if len(gen_outputs[j]) > t5_max_seq_len:
                                    continue

                                x = x.strip()
                                fantasy_prompt += 'Input\n\n' + x + '\n\n'
                                fantasy_prompt += 'Output\n\n' + gen_outputs[j] + '\n\n'
                                num_fantasy_examples += 1

                            # prepare fantasy items and add to fantasy dataset
                            if num_fantasy_examples > 0:
                                item = {}
                                item['prompt'] = fantasy_prompt 
                                item['code'] = edited_code 
                                item['inputs'] = data['inputs']
                                item['outputs'] = gen_outputs
                                edit_fantasy_dataset.append(item)
                                pbar2.update(1)

                                # for visualizing fantasy
                                if edit_fantasy_visualize_text == '':
                                    edit_fantasy_visualize_text += '\n' + 'original prompt' + ':\n\n' + data['prompt'] + '\n' 
                                    edit_fantasy_visualize_text += '-' * 10
                                    edit_fantasy_visualize_text += '\n' + 'fantasy prompt' + ':\n\n' + item['prompt'] + '\n' 
                                    edit_fantasy_visualize_text += '-' * 10
                                    edit_fantasy_visualize_text += '\n' + 'original code' + ':\n\n' + data['code'] + '\n' 
                                    edit_fantasy_visualize_text += '-' * 10
                                    edit_fantasy_visualize_text += '\n' + 'generated code' + ':\n\n' + item['code'] + '\n' 
                                    edit_fantasy_visualize_text += '-' * 10
        pbar2.close()

        results_edit_fantasy_fraction.append(len(edit_fantasy_dataset))

        all_edit_fantasy_dataset += edit_fantasy_dataset
        all_edit_fantasy_dataset = all_edit_fantasy_dataset[-all_edit_fantasy_buffer_size:]

        fsq.train() # put fsq in train mode for sleep training


    # create one (common) train dataset
    # train_dataset = train_dataset + new_train_dataset + all_edit_fantasy_dataset
    train_dataset = train_dataset + new_train_dataset

    # start FSQ training
    while train_step < num_train_steps + train_steps_done:

        # train on both train_dataset and fantasy_dataset
        sleep_on_what = -1
        minibatch = []

        if len(train_dataset) > 0:
            # fetch train minibatch
            idx = np.arange(len(train_dataset))
            np.random.shuffle(idx)
            idx = idx[:batch_size]
            minibatch += [train_dataset[i] for i in idx]

        if len(minibatch) > 0:
            sleep_on_what = 1

            if sleep_on_what != -1:

                # prepare program tokens
                program_batch = [item['code'] for item in minibatch]
                program_tokens_dict = t5_tokenizer(program_batch, return_tensors='pt', padding='max_length', truncation=True)
                # TODO use obtained attention mask for fsq encoder 
                program_tokens, program_attn_mask = program_tokens_dict.input_ids, program_tokens_dict.attention_mask

                # use bfloat16 precision for speed up # NOTE RoPE gives wrong results with bfloat16
                # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

                # forward prop through FSQ 
                recon_program_scores, z_e, z_q, usage, unique, percent_masked_latents = fsq(program_tokens.to(device)) # recon_programs.shape: [b, seq_len, num_levels]

                # calculate loss
                recon_loss = reconstruction_factor * reconstruction_loss(recon_program_scores, program_tokens.to(device))
                compress_loss = compress_factor * compression_loss(z_e)
                loss = recon_loss + compress_loss 

                # adjustment for gradient accumulation 
                loss_scaled = loss / gradient_accumulation_steps
                loss_scaled.backward()

                if (train_step + 1) % gradient_accumulation_steps == 0:
                    # gradient cliping - helps to prevent unnecessary divergence 
                    torch.nn.utils.clip_grad_norm_(sleep_params, max_norm=1.0)
                    # gradient step
                    sleep_optimizer.step()
                    sleep_optimizer.zero_grad()

                # bookeep losses
                results_train_loss = ema(results_train_loss, loss.item())
                results_codebook_usage.append(usage.item())
                results_codebook_unique.append(unique)
                results_masked_latents.append(percent_masked_latents.item())
                results_recon_loss.append(recon_loss.item())
                results_compress_loss.append(compress_loss.item())

                if len(results_train_loss) > 0:
                    pbar.set_description('fsq_loss: {:.3f}'.format(results_train_loss[-1]))

        pbar.update(1)


        ## plotting
        if (train_step+1) % plot_freq == 0: ## save ckpt and plot losses

            # save train buffer and fsq ckpt 
            with open(train_buffer_ckpt_path, 'wb') as f:
                pickle.dump(train_dataset, f)
            save_ckpt(device, fsq_ckpt_path, fsq, sleep_optimizer)

            # plot sleep results
            if len(results_train_loss) > 0:

                fig, ax = plt.subplots(2,2, figsize=(15,10))

                ax[0,0].plot(results_recon_loss, label='recon_loss')
                ax[0,0].plot(results_compress_loss, label='compress_loss')
                ax[0,0].plot(results_train_loss, label='train_loss')
                ax[0,0].legend()
                ax[0,0].set(xlabel='eval_iters')
                ax[0,0].set_title('train:{:.3f} recon:{:.3f} compress:{:.3f}'.format(results_train_loss[-1], results_recon_loss[-1], results_compress_loss[-1]))
                ax[0,0].set_ylim([0, 10])

                ax[1,0].plot(results_codebook_unique, label='codebook_unique')
                ax[1,0].legend()
                ax[1,0].set(xlabel='eval_iters')
                ax[1,0].set_title('val:{:.3f}'.format(results_codebook_unique[-1]))

                ax[0,1].plot(results_codebook_usage, label='codebook_usage')
                ax[0,1].legend()
                ax[0,1].set(xlabel='train_iters')
                ax[0,1].set_title('val:{:.3f}'.format(results_codebook_usage[-1]))

                ax[1,1].plot(results_masked_latents, label='percent_masked_latents')
                ax[1,1].legend()
                ax[1,1].set(xlabel='train_iters')
                ax[1,1].set_title('val:{:.3f}'.format(results_masked_latents[-1]))

                # plt.suptitle('final_train_loss: ' + str(results_train_loss[-1]))
                save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(1) + '.png'
                plt.savefig(save_path)
                plt.close(fig)


        train_step += 1

    pbar.close()