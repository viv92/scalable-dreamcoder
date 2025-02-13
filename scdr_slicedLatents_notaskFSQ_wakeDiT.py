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

    # hyperparams for vocab and bottleneck dims 
    dit_vocab_size = 1
    for n in num_quantized_values:
        dit_vocab_size *= n # equal to latent codebook size
    dit_vocab_size += 1 # for mask token
    dit_mask_token = dit_vocab_size - 1

    # hyperparams for latent slicing 
    program_latent_seqlen = 128 # 64 # 16 # 8 # 32 

    # hyperparams for fsq 
    fsq_d_model = 512 
    fsq_n_heads = 8
    assert fsq_d_model % fsq_n_heads == 0
    fsq_d_k = fsq_d_model // fsq_n_heads 
    fsq_d_v = fsq_d_k 
    fsq_n_layers = 6 # 3
    fsq_d_ff = fsq_d_model * 4

    # hyperparams for dit 
    dit_d_model = 512 
    dit_n_heads = 8
    assert dit_d_model % dit_n_heads == 0
    dit_d_k = dit_d_model // dit_n_heads
    dit_d_v = dit_d_k 
    dit_n_layers = 6 
    dit_d_ff = dit_d_model * 4

    dropout = 0.1 
    weight_decay = 0.1 
    compress_factor = 0.01 # 0 # 0.1 # 1
    reconstruction_factor = 1
    wake_factor = 0 # 1 

    # hyperparams for sleep mode
    sleep_mode = 1 # 0 = wake, 1 = sleep, 2 = dream    
    sleep_steps = 100 # 1000
    wake_steps = 20 # 100
    dream_steps = 100 # 1000  
    num_switches = -1
    sleep_steps_list = [wake_steps, sleep_steps, dream_steps]

    # hyperparams for training 
    diffusion_start_time_eps = 1e-3
    batch_size = 8 # 16 # 32
    gradient_accumulation_steps = 1 
    lr = 3e-4
    num_cycles = 20000
    num_train_steps = sum(sleep_steps_list) * num_cycles
    train_steps_done = 915200
    random_seed = 10
    resume_training_from_ckpt = True          

    # hyperparams for figures and plotting
    sampling_freq = (720 * 1) - 1 
    plot_freq = sampling_freq * 4

    # hyperparams for fantasies
    fantasy_batch_size = batch_size
    # fantasy_tries = 100 # 16 # 64
    # max_fantasies = fantasy_batch_size * fantasy_tries # number of fantasies added in one infusion
    max_correct_fantasies = 500 # 1000 # 10
    num_edit_fantasies = 1000 # 100
    all_edit_fantasy_buffer_size = 10 ** 3 # 2
    train_buffer_size = 10 ** 5 # 3
    delay_start_iter = -1 # sum(sleep_steps_list)
    fantasy_gen_cycle_counter = -1
    fantasy_gen_skip_cycles = 1 

    sampling_eps = 0.1 # 0.3 # 0
    sleep_loss_threshold = 0.3 # 0.1 # 0.2
    dream_loss_threshold = 2.5 # 8 # 6
    timeout = 60 * 60 * 24 # 1
    fant_timeout = 60 * 15 # 5
    init_train_dataset_size = 18500 # 100 # 10 # 1000
    infuse_items = 100 # 10 # 1000
    infuse_thresh = 100 # 10 # 1000

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # hyperparams for sampling
    num_sampling_steps = int(program_latent_seqlen * 0.25)
    wake_batch_size = batch_size
    p_uncond = 0.1 
    cfg_scale = 2.0

    # create hyperparam str
    hyperparam_dict = {}
    hyperparam_dict['method'] = 'scdr_notaskFSQ_fantAppend_FsqDitThresh_infuse' 
    hyperparam_dict['fsqD'] = fsq_d_model
    hyperparam_dict['fsqL'] = fsq_n_layers
    hyperparam_dict['pLatSeq'] = program_latent_seqlen
    hyperparam_dict['B'] = batch_size 
    # hyperparam_dict['lr'] = lr
    # hyperparam_dict['Wdecay'] = weight_decay
    # hyperparam_dict['drop'] = dropout
    hyperparam_dict['trainSz'] = init_train_dataset_size 
    hyperparam_dict['CF'] = compress_factor
    # hyperparam_dict['RF'] = reconstruction_factor
    # hyperparam_dict['WF'] = wake_factor
    # hyperparam_dict['initMode'] = sleep_mode
    # hyperparam_dict['sleep'] = sleep_steps
    hyperparam_dict['infuseN'] = infuse_items
    hyperparam_dict['infuseTh'] = infuse_thresh
    # hyperparam_dict['sampSteps'] = num_sampling_steps
    # hyperparam_dict['delay'] = delay_start_iter 
    # hyperparam_dict['Fnum'] = max_fantasies 
    hyperparam_dict['Fmax'] = max_correct_fantasies
    hyperparam_dict['sampEps'] = sampling_eps
    # hyperparam_dict['Fskips'] = fantasy_gen_skip_cycles 
    hyperparam_dict['editFbuf'] = all_edit_fantasy_buffer_size 
    hyperparam_dict['trainbuf'] = train_buffer_size 
    hyperparam_dict['swS'] = 0.1 # sleep_loss_threshold 
    hyperparam_dict['swD'] = dream_loss_threshold
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
    # test_dataset += train_dataset[init_train_dataset_size:]
    test_dataset = train_dataset[init_train_dataset_size:]
    train_dataset = train_dataset[:init_train_dataset_size]

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

    # init dit 
    dit_x_seq_len = program_latent_seqlen
    dit_max_seq_len = dit_x_seq_len + 1 # [t, x]
    dit_condition_dim = t5_d_model
    dit = init_dit(dit_max_seq_len, dit_x_seq_len, dit_d_model, dit_condition_dim, dit_vocab_size, dit_d_k, dit_d_v, dit_n_heads, dit_n_layers, dit_d_ff, dropout, device).to(device)

    # init optimizers
    sleep_params = fsq.parameters()
    sleep_optimizer = torch.optim.AdamW(sleep_params, lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)
    
    dream_params = dit.parameters()
    dream_optimizer = torch.optim.AdamW(dream_params, lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)

    # init sandbox 
    sandbox = Sandbox(default_timeout=2)

    # load ckpt
    if resume_training_from_ckpt:
        fsq, sleep_optimizer = load_ckpt(fsq_ckpt_path, fsq, sleep_optimizer, device=device, mode='train')
        dit, dream_optimizer = load_ckpt(dit_ckpt_path, dit, dream_optimizer, device=device, mode='train')
        with open(train_buffer_ckpt_path, 'rb') as f:
            train_dataset = pickle.load(f)

    # results for plotting
    results_test_items = [len(test_dataset)]
    results_dit_loss, results_dit_success_accuracy, results_dit_correct_accuracy, results_fantasy_fraction, results_fantasy_correct_fraction, results_wake_dit_loss, results_edit_fantasy_fraction = [], [], [], [], [], [], []
    results_train_loss, results_recon_loss, results_compress_loss = [], [], []
    results_codebook_usage, results_codebook_unique, results_masked_latents = [], [], []
    results_infuse_wake, results_infuse_fant = [], []

    # train

    train_step = train_steps_done
    sleep_mode_counter = 0 
    pbar = tqdm(total=num_train_steps)

    solved_test_idx = [] # used to remove solved test items from test dataset

    new_train_dataset = []
    # new_train_dataset = train_dataset
    # train_dataset = []

    new_fantasy_dataset = []
    new_fantasy_correct_dataset = []
    all_edit_fantasy_dataset = [] 
    edit_fantasy_dataset = []
    wake_visualize_text = ''
    fantasy_visualize_text = ''
    edit_fantasy_visualize_text = ''
    infuse = 0
    start_time = time()
    
    while train_step < num_train_steps + train_steps_done:

        # handle sleep mode 
        if not (num_switches == 0):
            switch_required = 0

            # wake to sleep
            if (sleep_mode == 0) and (sleep_mode_counter == sleep_steps_list[sleep_mode]): # wake to sleep
                switch_required = 1
                start_time = time()

            # sleep to dream
            if (sleep_mode == 1):
                if time() - start_time > timeout:
                    switch_required = 1
                    start_time = time()

                    # save train buffer and fsq ckpt 
                    with open(train_buffer_ckpt_path, 'wb') as f:
                        pickle.dump(train_dataset, f)
                    save_ckpt(device, fsq_ckpt_path, fsq, sleep_optimizer)
                    
                else:
                    if (sleep_mode_counter > sleep_steps_list[sleep_mode]) and (results_train_loss[-1] < sleep_loss_threshold): 
                        switch_required = 1
                        start_time = time()
                        # sleep_loss_threshold = results_train_loss[-1] * 0.95 # 0.99

                        # save train buffer and fsq ckpt 
                        with open(train_buffer_ckpt_path, 'wb') as f:
                            pickle.dump(train_dataset, f)
                        save_ckpt(device, fsq_ckpt_path, fsq, sleep_optimizer)


            # dream to wake
            if (sleep_mode == 2):
                if time() - start_time > timeout:
                    switch_required = 1

                    # save dit ckpt 
                    save_ckpt(device, dit_ckpt_path, dit, dream_optimizer)
                    
                else:
                    if (sleep_mode_counter > sleep_steps_list[sleep_mode]) and (results_dit_loss[-1] < dream_loss_threshold): 
                        switch_required = 1
                        # dream_loss_threshold = results_dit_loss[-1] * 0.95 # 0.99

                        # save dit ckpt 
                        save_ckpt(device, dit_ckpt_path, dit, dream_optimizer)


            # if switch from wake to sleep, remove test items solved in wake phase
            if switch_required and (sleep_mode == 0):
                # solved_test_idx = list(set(solved_test_idx)) # deduplicate
                # test_dataset = [x for i,x in enumerate(test_dataset) if i not in solved_test_idx]
                # solved_test_idx = []
                remaining_test_items = len(test_dataset)
                results_test_items.append(remaining_test_items)
                results_infuse_wake.append(20000 - infuse * 10)

            # if switch from dream to wake, shift new_train_data to train_data
            if switch_required and (sleep_mode == 2):
                train_dataset += new_train_dataset
                # if len(edit_fantasy_dataset) > 0:
                #     train_dataset += edit_fantasy_dataset[-100:]
                train_dataset = train_dataset[-train_buffer_size:]
                new_train_dataset = []
                # shift new_fantasy_correct_data to new_train_data
                # train_dataset += new_fantasy_dataset
                # new_fantasy_dataset = []

            while switch_required:
                sleep_mode += 1
                sleep_mode = sleep_mode % len(sleep_steps_list)
                sleep_mode_counter = 0
                switch_required = (sleep_mode_counter == sleep_steps_list[sleep_mode]) 
                num_switches -= 1


        if (sleep_mode == 0) and (len(test_dataset) > 0): # wake mode - get solved test data 
            dit.eval()
            fsq.eval()

            # fetch test minibatch
            test_idx = np.arange(len(test_dataset))
            np.random.shuffle(test_idx)
            test_idx = test_idx[:wake_batch_size]
            minibatch = [test_dataset[i] for i in test_idx]

            with torch.no_grad():

                # get caption embeddings
                cap_batch = [item['prompt'] for item in minibatch]
                cap_tokens_dict = t5_tokenizer(cap_batch, return_tensors='pt', padding=True, truncation=True)
                cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
                cap_embs = t5_model.encoder(input_ids=cap_tokens.to(device), attention_mask=cap_attn_mask.to(device)).last_hidden_state 

                # get sample tokens corresponding to indices of codebook 
                x_sample = get_sample(dit, program_latent_seqlen, dit_mask_token, dit_vocab_size, num_sampling_steps, cap_embs.shape[0], cap_embs, cfg_scale, 0, device) # x_sample.shape: [b, seqlen]
                x_sample = x_sample.flatten() # shape: [b * seqlen]

                # get codebook vectors indexed by x_sample
                sampled_latents = fsq.codebook[x_sample] # sampled_latents.shape: [b * program_latent_seqlen, latent_dim]
                sampled_latents = sampled_latents.unflatten(dim=0, sizes=(len(minibatch), program_latent_seqlen)) # [b, program_latent_seqlen, latent_dim]
                padded_latents = torch.zeros(sampled_latents.shape[0], t5_max_seq_len, sampled_latents.shape[-1], device=sampled_latents.device)
                padded_latents[:, :sampled_latents.shape[1]] = sampled_latents
                padded_latents = padded_latents.flatten(start_dim=0, end_dim=1) # [b * program_seq_len, latent_dim]
                pred_program_scores, _ = fsq.decode(padded_latents.float()) # [b, seqlen, num_levels]
                pred_program_tokens = torch.argmax(pred_program_scores, dim=-1).long() # [b, seqlen]

                # decode programs
                pred_programs = t5_tokenizer.batch_decode(pred_program_tokens, skip_special_tokens=True)

                # execute programs 
                solved_minibatch_idx = []
                for i in range(len(minibatch)):
                    data = minibatch[i]
                    program = pred_programs[i]
                    success = 1
                    correct = 1
                    for j, x in enumerate(data['inputs']):
                        # format input correctly
                        x = x.strip()
                        x = x.splitlines()
                        # execute
                        output, error = sandbox.execute(program, inputs=x)
                        if error:
                            success *= 0
                            break 
                        else: # check for correctness 
                            gt_output = data['outputs'][j].strip()
                            if (output is not None) and (output.strip() != ''):
                                output = output.strip()
                            eq = output == gt_output
                            correct *= eq 
                    if (len(data['inputs']) > 0) and success:
                        if correct:
                            solved_minibatch_idx.append(i)

            # add solved test items to new train dataset (including solving program)
            for i in solved_minibatch_idx:
                # if wake_visualize_text == '':
                item = minibatch[i] 
                wake_visualize_text += '\n' + 'prompt' + ':\n\n' + item['prompt'] + '\n' 
                wake_visualize_text += '-' * 10
                wake_visualize_text += '\n' + 'original code' + ':\n\n' + item['code'] + '\n' 
                wake_visualize_text += '-' * 10
                wake_visualize_text += '\n' + 'generated code' + ':\n\n' + pred_programs[i] + '\n' 
                wake_visualize_text += '-' * 10

                test_index = test_idx[i]
                test_item = test_dataset[test_index]
                test_item['code'] = pred_programs[i] # [seqlen]
                new_train_dataset.append(test_item)
                solved_test_idx.append(test_index) # to later remove the solved test items from test dataset
                
            test_dataset = [x for i,x in enumerate(test_dataset) if i not in solved_test_idx]
            solved_test_idx = []

            # wake DiT training 
            dit.train()

            # x = x_sample.view(len(minibatch), program_latent_seqlen) # x.shape: [b, seq_len] 
            # condition = cap_embs

            # # set condition = None with prob p_uncond
            # if np.random.rand() < p_uncond: # TODO: explore the effect of no CFG versus CFG only during training versus CFG during training and sampling
            #     condition = None

            # # sample diffusion time ~ uniform(eps, 1)
            # t = (1 - diffusion_start_time_eps) * torch.rand(x.shape[0], device=device) + diffusion_start_time_eps

            # # get noise from noise schedule
            # sigma, dsigma = logLinearNoise(t)

            # # perturb the data
            # x_perturb = perturb(x, sigma, dit_mask_token)

            # # use bfloat16 precision for speed up # NOTE RoPE gives wrong results with bfloat16
            # # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

            # # get score
            # log_score = dit(x_perturb, sigma, condition)

            # # calculate loss 
            # dit_loss = score_entropy_loss(log_score, sigma.unsqueeze(-1), x_perturb, x, dit_mask_token)
            # dit_loss = (dsigma.unsqueeze(-1) * dit_loss).sum(dim=-1)

            # # set signs according to solved 
            # dit_loss[solved_minibatch_idx] *= -1 
            # wake_dit_loss = -1 * wake_factor * dit_loss.mean()

            # wake_dit_loss.backward()

            # if sleep_mode_counter == sleep_steps_list[sleep_mode] - 1: # take one gradient step
            #     # gradient cliping - helps to prevent unnecessary divergence 
            #     torch.nn.utils.clip_grad_norm_(dream_params, max_norm=1.0)
            #     # gradient step
            #     dream_optimizer.step()
            #     dream_optimizer.zero_grad()

            # # bookeep losses
            # results_wake_dit_loss = ema(results_wake_dit_loss, wake_dit_loss.item())

            pbar.update(1)
            # pbar.set_description('mode:{} test_items:{} wake_dit_loss:{:.3f}'.format(sleep_mode, len(test_dataset) - len(new_train_dataset), results_wake_dit_loss[-1]))
            # pbar.set_description('mode:{} test_items:{}'.format(sleep_mode, len(test_dataset) - len(new_train_dataset)))
            pbar.set_description('mode:{} test_items:{}'.format(sleep_mode, len(test_dataset)))
            fsq.train() 


        ## generate fantasy dataset 
        if (sleep_mode == 1) and (sleep_mode_counter == 0) and (train_step > delay_start_iter):
            fsq.eval() 
            dit.eval()
            fantasy_gen_cycle_counter += 1

            if (fantasy_gen_cycle_counter > 0) and (fantasy_gen_cycle_counter % fantasy_gen_skip_cycles == 0):
                with torch.no_grad():

                    # generate fantasies using DiT

                    pbar2 = tqdm(total=max_correct_fantasies)
                    fant_start_time = time()

                    seen_fantasy_code = set() # to avoid duplicate fantasies

                    # for j in range(fantasy_tries):
                    while len(new_fantasy_correct_dataset) < max_correct_fantasies:

                        pbar2.set_description('Generating fantasies - fant:{} correct:{}'.format(len(new_fantasy_dataset), len(new_fantasy_correct_dataset)))

                        if time() - fant_start_time > fant_timeout: 
                            break 

                        # fetch test minibatch
                        test_idx = np.arange(len(test_dataset))
                        np.random.shuffle(test_idx)
                        test_idx = test_idx[:fantasy_batch_size]
                        minibatch = [test_dataset[i] for i in test_idx]

                        with torch.no_grad():

                            # get caption embeddings
                            cap_batch = [item['prompt'] for item in minibatch]
                            cap_tokens_dict = t5_tokenizer(cap_batch, return_tensors='pt', padding=True, truncation=True)
                            cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
                            cap_embs = t5_model.encoder(input_ids=cap_tokens.to(device), attention_mask=cap_attn_mask.to(device)).last_hidden_state 

                            # get sample tokens corresponding to indices of codebook 
                            x_sample = get_sample(dit, program_latent_seqlen, dit_mask_token, dit_vocab_size, num_sampling_steps, cap_embs.shape[0], cap_embs, cfg_scale, sampling_eps, device) # x_sample.shape: [b, seqlen]
                            x_sample = x_sample.flatten() # shape: [b * seqlen]

                            # get codebook vectors indexed by x_sample
                            sampled_latents = fsq.codebook[x_sample] # sampled_latents.shape: [b * program_latent_seqlen, latent_dim]
                            sampled_latents = sampled_latents.unflatten(dim=0, sizes=(len(minibatch), program_latent_seqlen)) # [b, program_latent_seqlen, latent_dim]
                            padded_latents = torch.zeros(sampled_latents.shape[0], t5_max_seq_len, sampled_latents.shape[-1], device=sampled_latents.device)
                            padded_latents[:, :sampled_latents.shape[1]] = sampled_latents
                            padded_latents = padded_latents.flatten(start_dim=0, end_dim=1) # [b * program_seq_len, latent_dim]
                            pred_program_scores, _ = fsq.decode(padded_latents.float()) # [b, seqlen, num_levels]
                            pred_program_tokens = torch.argmax(pred_program_scores, dim=-1).long() # [b, seqlen]

                            # decode programs
                            pred_programs = t5_tokenizer.batch_decode(pred_program_tokens, skip_special_tokens=True)

                            # execute programs 
                            solved_minibatch_idx = []
                            for i in range(len(minibatch)):
                                data = minibatch[i]
                                program = pred_programs[i]
                                success = 1
                                correct = 1
                                gen_outputs = []
                                for j, x in enumerate(data['inputs']):
                                    # format input correctly
                                    x = x.strip()
                                    x = x.splitlines()
                                    # execute
                                    output, error = sandbox.execute(program, inputs=x)
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
                                if (len(data['inputs']) > 0) and success:
                                    if correct: # correct - add to train set
                                        solved_minibatch_idx.append(i)
                                        pbar2.update(1)
                                    else: # incorrect - add to fantasy set
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
                                                item['code'] = program 
                                                item['inputs'] = data['inputs']
                                                item['outputs'] = gen_outputs

                                                if program not in seen_fantasy_code:
                                                    new_fantasy_dataset.append(item)
                                                    seen_fantasy_code.add(program)
                                                    # pbar2.update(1)

                                                # for visualizing fantasy
                                                # if fantasy_visualize_text == '':
                                                fantasy_visualize_text += '\n' + 'original prompt' + ':\n\n' + data['prompt'] + '\n' 
                                                fantasy_visualize_text += '-' * 10
                                                fantasy_visualize_text += '\n' + 'fantasy prompt' + ':\n\n' + item['prompt'] + '\n' 
                                                fantasy_visualize_text += '-' * 10
                                                fantasy_visualize_text += '\n' + 'original code' + ':\n\n' + data['code'] + '\n' 
                                                fantasy_visualize_text += '-' * 10
                                                fantasy_visualize_text += '\n' + 'generated code' + ':\n\n' + item['code'] + '\n' 
                                                fantasy_visualize_text += '-' * 10


                        # add solved test items to fantasy_correct_dataset
                        for i in solved_minibatch_idx:
                            # if wake_visualize_text == '':
                            item = minibatch[i] 
                            wake_visualize_text += '\n' + 'prompt' + ':\n\n' + item['prompt'] + '\n' 
                            wake_visualize_text += '-' * 10
                            wake_visualize_text += '\n' + 'original code' + ':\n\n' + item['code'] + '\n' 
                            wake_visualize_text += '-' * 10
                            wake_visualize_text += '\n' + 'generated code' + ':\n\n' + pred_programs[i] + '\n' 
                            wake_visualize_text += '-' * 10

                            test_index = test_idx[i]
                            test_item = test_dataset[test_index]
                            test_item['code'] = pred_programs[i] # [seqlen]
                            new_fantasy_correct_dataset.append(test_item)
                            solved_test_idx.append(test_index) # to later remove the solved test items from test dataset

                        test_dataset = [x for i,x in enumerate(test_dataset) if i not in solved_test_idx]
                        solved_test_idx = []

                    pbar2.close()
                    results_fantasy_fraction.append(len(new_fantasy_dataset))
                    results_fantasy_correct_fraction.append(len(new_fantasy_correct_dataset))

                    # shift fantasy_correct_dataset to new_train_dataset and update solved test tasks
                    if len(new_fantasy_correct_dataset) > 0:
                        new_train_dataset += new_fantasy_correct_dataset
                        new_fantasy_correct_dataset = []
                        # solved_test_idx = list(set(solved_test_idx)) # deduplicate
                        # test_dataset = [x for i,x in enumerate(test_dataset) if i not in solved_test_idx]
                        # solved_test_idx = []

                # infusion 
                if (len(new_train_dataset) < infuse_thresh) and (results_train_loss[-1] < sleep_loss_threshold) and (results_dit_loss[-1] < dream_loss_threshold):
                    infuse = infuse_items
                    new_train_dataset += test_dataset[:infuse]
                    test_dataset = test_dataset[infuse:]
                else:
                    infuse = 0
                results_infuse_fant.append(infuse)


            # generate fantasies by editing train programs

            edit_fantasy_dataset = []
            pbar2 = tqdm(total=num_edit_fantasies)
            fant_start_time = time()

            while len(edit_fantasy_dataset) < num_edit_fantasies:

                pbar2.set_description('Generating edit fantasies')

                if time() - fant_start_time > fant_timeout: 
                    break 

                # fetch train minibatch
                train_idx = np.arange(len(train_dataset))
                np.random.shuffle(train_idx)
                train_idx = train_idx[:fantasy_batch_size]
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
            # shift new fantasy dataset to edit fantasy dataset
            all_edit_fantasy_dataset += new_fantasy_dataset
            new_fantasy_dataset = []

            fsq.train() # put fsq in train mode for sleep training


        if sleep_mode == 1: # sleep mode - train FSQ 

            # train on both train_dataset and fantasy_dataset
            sleep_on_what = -1
            minibatch = []

            if len(train_dataset) > 0:
                # fetch train minibatch
                idx = np.arange(len(train_dataset))
                np.random.shuffle(idx)
                idx = idx[:batch_size]
                minibatch += [train_dataset[i] for i in idx]

            if len(new_train_dataset) > 0:
                # fetch new train minibatch
                idx = np.arange(len(new_train_dataset))
                np.random.shuffle(idx)
                idx = idx[:batch_size]
                minibatch += [new_train_dataset[i] for i in idx]

            # if len(new_fantasy_dataset) > 0:
            #     # fetch new fantasy minibatch
            #     idx = np.arange(len(new_fantasy_dataset))
            #     np.random.shuffle(idx)
            #     idx = idx[:batch_size]
            #     minibatch += [new_fantasy_dataset[i] for i in idx]

            if len(all_edit_fantasy_dataset) > 0:
                # fetch edit fantasy minibatch
                idx = np.arange(len(all_edit_fantasy_dataset))
                np.random.shuffle(idx)
                idx = idx[:batch_size]
                minibatch += [all_edit_fantasy_dataset[i] for i in idx]

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

                    loss.backward()
                    # gradient cliping 
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
                        pbar.set_description('mode:{} loss: {:.3f}'.format(sleep_mode, results_train_loss[-1]))

            pbar.update(1)
            dit.train()


        if sleep_mode == 2: # dream mode - train DiT
            fsq.eval()

            ## dream mode training

            # train on both replays and fantasies 
            dream_on_what = -1
            minibatch = []

            if len(train_dataset) > 0:
                # fetch train minibatch
                idx = np.arange(len(train_dataset))
                np.random.shuffle(idx)
                idx = idx[:batch_size]
                minibatch += [train_dataset[i] for i in idx]

            if len(new_train_dataset) > 0:
                # fetch new train minibatch
                idx = np.arange(len(new_train_dataset))
                np.random.shuffle(idx)
                idx = idx[:batch_size]
                minibatch += [new_train_dataset[i] for i in idx]

            # if len(new_fantasy_dataset) > 0:
            #     # fetch new fantasy minibatch
            #     idx = np.arange(len(new_fantasy_dataset))
            #     np.random.shuffle(idx)
            #     idx = idx[:batch_size]
            #     minibatch += [new_fantasy_dataset[i] for i in idx]

            if len(all_edit_fantasy_dataset) > 0:
                # fetch edit fantasy minibatch
                idx = np.arange(len(all_edit_fantasy_dataset))
                np.random.shuffle(idx)
                idx = idx[:batch_size]
                minibatch += [all_edit_fantasy_dataset[i] for i in idx]

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

                    # forward prop through fsq encoder to get target latents
                    with torch.no_grad():
                        z_e = fsq.encode(program_tokens.to(device)) # z_e.shape: [b, seq_len,  img_latent_dim]
                        latents, _, _, _, _, target_idx = fsq.quantize(z_e) # target_idx.shape: [b * img_latent_seqlen]
                        target_idx = target_idx.view(len(minibatch), t5_max_seq_len) # [b, seqlen] 

                    target_idx = target_idx[:, :program_latent_seqlen] # program latent slicing

                    x = target_idx # x.shape: [b, seq_len] 
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

                    dream_loss.backward()
                    # gradient cliping - helps to prevent unnecessary divergence 
                    torch.nn.utils.clip_grad_norm_(dream_params, max_norm=1.0)
                    # gradient step
                    dream_optimizer.step()
                    dream_optimizer.zero_grad()

                    # bookeep losses
                    results_dit_loss = ema(results_dit_loss, dream_loss.item()) 

                    if len(results_dit_loss) > 0:
                        pbar.set_description('mode:{} loss: {:.2f}'.format(sleep_mode, results_dit_loss[-1]))

            pbar.update(1)

            fsq.train()


        ## sampling
        # if (train_step+1) % sampling_freq == 0:
        if sleep_mode_counter == 1:

            # if sleep_mode == 1: # sleep mode - eval FSQ 

            #     # visualize reconstructions
            #     if sleep_on_what != -1:

            #         example = minibatch[0]['examples'][0]
            #         example_input, example_output = example[:(square_dim**2)], example[-(square_dim**2):] # [seqlen]

            #         recon_program_scores = recon_program_scores[0].detach() # [seqlen, num_levels]
            #         pred_program_weights = torch.argmax(recon_program_scores, dim=-1).long() # [seqlen]

            #         # forward prop through program net to get pred outputs 
            #         set_weights(program_net, pred_program_weights, pn_interval, pn_lower_limit)
            #         pred_output = program_net.predict(example_input.unsqueeze(0).to(device)).long() # [1, seqlen]
            #         pred_output = pred_output.squeeze(0).cpu() # [seqlen]

            #         vis_item = {}
            #         vis_item['pred_answer'] = pred_output 
            #         vis_item['query'] = example_input 
            #         vis_item['answer'] = example_output 
            #         vis_item['qid'] = minibatch[0]['qid']
            #         savepath = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(1)
            #         visualize_grids([vis_item], savepath, square_dim)


            if sleep_mode == 2: # dream mode - eval DIT 
            
                # put model in eval mode to avoid dropout
                dit.eval()
                fsq.eval()

                # generate sample from dit
                # if dream_on_what != -1:
                if sleep_on_what != -1:

                    with torch.no_grad():

                        # get caption embeddings
                        cap_batch = [item['prompt'] for item in minibatch]
                        cap_tokens_dict = t5_tokenizer(cap_batch, return_tensors='pt', padding=True, truncation=True)
                        cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
                        cap_embs = t5_model.encoder(input_ids=cap_tokens.to(device), attention_mask=cap_attn_mask.to(device)).last_hidden_state 

                        # get sample tokens corresponding to indices of codebook 
                        x_sample = get_sample(dit, program_latent_seqlen, dit_mask_token, dit_vocab_size, num_sampling_steps, cap_embs.shape[0], cap_embs, cfg_scale, 0, device) # x_sample.shape: [b, seqlen]
                        x_sample = x_sample.flatten() # shape: [b * seqlen]

                        # get codebook vectors indexed by x_sample
                        sampled_latents = fsq.codebook[x_sample] # sampled_latents.shape: [b * program_latent_seqlen, latent_dim]
                        sampled_latents = sampled_latents.unflatten(dim=0, sizes=(len(minibatch), program_latent_seqlen)) # [b, program_latent_seqlen, latent_dim]
                        padded_latents = torch.zeros(sampled_latents.shape[0], t5_max_seq_len, sampled_latents.shape[-1], device=sampled_latents.device)
                        padded_latents[:, :sampled_latents.shape[1]] = sampled_latents
                        padded_latents = padded_latents.flatten(start_dim=0, end_dim=1) # [b * program_seq_len, latent_dim]
                        pred_program_scores, _ = fsq.decode(padded_latents.float()) # [b, seqlen, num_levels]
                        pred_program_tokens = torch.argmax(pred_program_scores, dim=-1).long() # [b, seqlen]

                        # decode programs
                        pred_programs = t5_tokenizer.batch_decode(pred_program_tokens, skip_special_tokens=True)

                        # execute programs 
                        success_runs = 0
                        correct_runs = 0
                        for i in range(len(minibatch)):
                            data = minibatch[i]
                            program = pred_programs[i]
                            success = 1
                            correct = 1
                            for j, x in enumerate(data['inputs']):
                                # format input correctly
                                x = x.strip()
                                x = x.splitlines()
                                # execute
                                output, error = sandbox.execute(program, inputs=x)
                                if error:
                                    success *= 0
                                    break 
                                else: # check for correctness 
                                    gt_output = data['outputs'][j].strip()
                                    if (output is not None) and (output.strip() != ''):
                                        output = output.strip()
                                    eq = output == gt_output
                                    correct *= eq 
                            if (len(data['inputs']) > 0) and success:
                                success_runs += 1
                                if correct:
                                    correct_runs += 1

                        # get solved minibatch idx
                        success_accuracy = success_runs / len(minibatch)
                        correct_accuracy = correct_runs / len(minibatch)
                        results_dit_success_accuracy.append(success_accuracy)
                        results_dit_correct_accuracy.append(correct_accuracy)

                dit.train()
                fsq.train()


        ## plotting
        # if (train_step+1) % plot_freq == 0: ## save ckpt and plot losses
        if sleep_mode_counter == 1:

            if sleep_mode == 0: # wake mode - plot test items and visualize code

                if len(results_test_items) > 0:
                    fig = plt.figure()
                    plt.plot(results_test_items, label='test_items')
                    plt.plot(results_infuse_wake, label='infusion')
                    plt.legend()
                    plt.title('val:{}'.format(results_test_items[-1]))
                    save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(0) + '_remItems.png'
                    fig.savefig(save_path)
                    plt.close(fig)

                if len(results_wake_dit_loss) > 0:
                    fig = plt.figure()
                    plt.plot(results_wake_dit_loss, label='wake_dit_loss')
                    plt.legend()
                    plt.title('val:{}'.format(results_wake_dit_loss[-1]))
                    save_path = results_dir + 'plot_trainStep=' + str(train_step) + '_sleepMode=' + str(0) + '_wakeDitLoss.png'
                    fig.savefig(save_path)
                    plt.close(fig)

                if wake_visualize_text != '':
                    save_path = results_dir + 'code_trainStep=' + str(train_step) + '.txt'
                    with open(save_path, 'w') as f:
                        f.write(wake_visualize_text)
                    wake_visualize_text = ''

                if fantasy_visualize_text != '':
                    save_path = results_dir + 'fantasy_code_trainStep=' + str(train_step) + '.txt'
                    with open(save_path, 'w') as f:
                        f.write(fantasy_visualize_text)
                    fantasy_visualize_text = ''

                if edit_fantasy_visualize_text != '':
                    save_path = results_dir + 'edit_fantasy_code_trainStep=' + str(train_step) + '.txt'
                    with open(save_path, 'w') as f:
                        f.write(edit_fantasy_visualize_text)
                    edit_fantasy_visualize_text = ''


            if sleep_mode == 1: # sleep mode - plot for FSQ

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


            if sleep_mode == 2: # 2: # dream mode - plot for DIT 

                # plot dit loss
                if len(results_dit_loss) > 0:

                    fig = plt.figure()
                    plt.plot(results_dit_loss, label='dit_loss')
                    plt.legend()
                    plt.title('final_loss:{:.3f}'.format(results_dit_loss[-1]))
                    plt.ylim([0, 100])
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
        sleep_mode_counter += 1

    pbar.close()