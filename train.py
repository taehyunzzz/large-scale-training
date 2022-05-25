import argparse
import os
import shutil
import time
import copy
import json

import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch import nn, Tensor

import torchtext
from torchtext.datasets import WikiText2
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import transformers

import sys
np.set_printoptions(threshold=sys.maxsize)

import deepspeed
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from deepspeed.runtime.utils import see_memory_usage
from deepspeed import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig, DeepSpeedConfig

print("PATH {}".format(sys.path))
print("Using deepspeed from {}".format(deepspeed.__file__))

from pynvml import *

import math
from typing import Tuple

import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

import torch.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity

# limit use of memory (fraction from total memory, device id)
#torch.cuda.set_per_process_memory_fraction(0.08, 0)
total_memory = torch.cuda.get_device_properties(0).total_memory

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TransformerModel(nn.Module):
    
    def __init__(self, cuda_config, ntoken: int, d_model: int, nhead: int, d_hid: int,
            nlayers: int, dropout: float = 0.5):

        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        

        # encoder_layers = TransformerEncoderLayer(d_model=d_model, 
        #                                             nhead=nhead, 
        #                                             # should be 4xd_model
        #                                             dim_feedforward=d_hid, 
        #                                             dropout=dropout)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_encoder = nn.ModuleList([
            copy.deepcopy(DeepSpeedTransformerLayer(cuda_config))
            for _ in range(nlayers)
        ])
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
        src: Tensor, shape [seq_len, batch_size]
        src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
        output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        for layer in self.transformer_encoder:
            src = layer(src, attention_mask=src_mask)
        output = self.decoder(src)
        return output

class TransformerModel2(nn.Module):

    def __init__(self, cuda_config):

        super().__init__()
        self.model_type = 'Transformer'

        self.layer = nn.ModuleList([
            copy.deepcopy(DeepSpeedTransformerLayer(cuda_config))
            for _ in range(cuda_config.num_hidden_layers)
        ])

    def forward(self, src: Tensor, **kwargs) -> Tensor:

        for layer_ in self.layer:
            print("Layer type is : {}".format(type(layer_)))
            if type(layer_) == "DeepSpeedTransformerLayer":
                src = layer_(src, attention_mask=kwargs["src_mask"])
            else :
                src = layer_(src)
        
        return src

# def generate_square_subsequent_mask(sz: int) -> Tensor:
#     """Generates an upper-triangular matrix of -inf, with zeros on diag."""
#     return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    # mask_1d = torch.ones(sz) * float('-inf')
    # mask_2d = torch.unsqueeze(mask_1d, dim=1)
    mask_2d = torch.ones(sz,sz) * float('-inf')
    mask_3d = torch.unsqueeze(mask_2d, dim=0)
    mask_4d = torch.unsqueeze(mask_3d, dim=0)
    return mask_4d

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
    data: Tensor, shape [N]
    bsz: int, batch size

    Returns:
    Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

def cpu_utilization():
    load1, _, _ = psutil.getloadavg()
    cpu_usage = (load1 / os.cpu_count()) * 100

    virt_mem = psutil.virtual_memory()
    mem_usage = virt_mem.percent

    return cpu_usage, mem_usage
    

def gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)

    return info.used//1024**2

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_batch(source: Tensor, i: int, bptt) -> Tuple[Tensor, Tensor]:
    """
    Args:
    source: Tensor, shape [full_seq_len, batch_size]
    i: int

    Returns:
    tuple (data, target), where data has shape [seq_len, batch_size] and
    target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--local_rank', default=0, type=int) 
    parser.add_argument('--master_port', default="8888", type=str) 

    parser.add_argument('--batch_size', default=1, type=int) 
    parser.add_argument('--bptt', default=10, type=int) 
    parser.add_argument('--emsize', default=768, type=int) 
    parser.add_argument('--layers', default=12, type=int) 
    parser.add_argument('--hidden', default=3072, type=int) 
    parser.add_argument('--heads', default=12, type=int) 
    parser.add_argument('--dropout', default=.1, type=float) 
    parser.add_argument('--dataset', default="wikitext2", type=str) 
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--resume', default=False) 
    parser.add_argument('--cpu_offload_param', default="0", type=str) 
    parser.add_argument('--cpu_offload_optim', default="0", type=str) 
    parser.add_argument('--suffix', default="", type=str) 
    parser.add_argument('--fp16', action="store_true")
    parser.add_argument('--no_write_results', action="store_true")

    return parser


def train(model : nn.Module, train_data, bptt, epoch=0) -> None :

    prep_time      = AverageMeter()
    forward_time   = AverageMeter()
    backward_time  = AverageMeter()
    update_time    = AverageMeter()
    throughput     = AverageMeter()
    cpu_util       = AverageMeter()
    cpu_mem_util   = AverageMeter()
    gpu_mem_util   = AverageMeter()

    model.train()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt

    flops, params = 0, 0

    exit_batch = 30

    if epoch == 1 :
        exit_batch = 3

    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):

        batch_timestamp = time.time()
        timestamp = batch_timestamp

        with record_function("get_data"):
            data, targets = get_batch(train_data, i, bptt)
        batch_size = data.size(0)

        if batch_size != bptt :
            src_mask = src_mask[:batch_size, :batch_size]

        prep_time.update(time.time() - timestamp)
        timestamp = time.time()

        with record_function("forward"):
            output = model(data, src_mask=src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        #if batch == exit_batch : see_memory_usage("before fwd", force=True)
        forward_time.update(time.time() - timestamp)
        timestamp = time.time()

        #optimizer.zero_grad()
        #loss.backward()
        #if batch == exit_batch : see_memory_usage("before bwd", force=True)

        with record_function("backward"):
            model.backward(loss)

        backward_time.update(time.time() - timestamp)
        timestamp = time.time()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        #if batch == exit_batch : see_memory_usage("before wup", force=True)

        with record_function("weight_update"):
            model.step()

        #if batch == exit_batch : see_memory_usage("after wup", force=True)
        #optimizer.step()

        update_time.update(time.time() - timestamp)

        cpu_usage, cpu_mem_usage = cpu_utilization()
        cpu_util.update(cpu_usage)
        cpu_mem_util.update(cpu_mem_usage)
        
        timestamp = time.time()

        # throughput in sequences per microsecond
        throughput.update(batch_size / float(time.time() - batch_timestamp), batch_size)

        if batch == exit_batch:

            #return prep_time.avg, forward_time.avg, backward_time.avg, update_time.avg, throughput.avg, cpu_util.avg, cpu_mem_util.avg
            return 0,0,0,0,0,0,0

    return prep_time.avg, forward_time.avg, backward_time.avg, update_time.avg, throughput.avg, cpu_util.avg, cpu_mem_util.avg
        
def evaluate(model: nn.Module, eval_data: Tensor) -> float:

    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            batch_size = data.size(0)

            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += batch_size * criterion(output_flat, targets).item()

    return total_loss / (len(eval_data) - 1)

def main():
    global args, vocab, ntokens, tokenizer, device, criterion, optimizer

    parser = parse()
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    cudnn.benchmark = True

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])

    else :
        args.gpu = args.local_rank
        args.world_size = 1

    if args.distributed:
        deepspeed.init_distributed(dist_backend='nccl')
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        args.world_size = torch.distributed.get_world_size()
        print("### global rank of curr node: {}".format(torch.distributed.get_rank()))

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.dataset == "wikitext2":
        train_iter = WikiText2(split='train')
    elif args.dataset == "penntreebank":
        train_iter = PennTreebank(split='train')

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    ntokens = len(vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    cuda_config = DeepSpeedTransformerConfig(batch_size = args.batch_size,
                                    # max_seq_length = 128,
                                    hidden_size = args.hidden,
                                    heads = args.heads,
                                    attn_dropout_ratio = 0.1,
                                    hidden_dropout_ratio = 0.1,
                                    num_hidden_layers = 1,
                                    initializer_range = 0.02,
                                    local_rank = 0,
                                    seed = 1234,
                                    fp16 = True,
                                    pre_layer_norm=True,
                                    attn_dropout_checkpoint=False,
                                    normalize_invertible=False,
                                    gelu_checkpoint=False)
    model = TransformerModel(   
                                cuda_config=cuda_config,
                                ntoken=ntokens, 
                                d_model=args.emsize,
                                nhead=args.heads,
                                d_hid=args.hidden, 
                                nlayers=args.layers, 
                                dropout=args.dropout
                            )
    #model = TransformerModel2(cuda_config)  
    
    #print("MODEL INFO\n{}".format(model))

    #print("Estimated memory requirements for deepspeed ZERO3\n")

    print("DS config file {}:".format(args.deepspeed_config))

    if args.deepspeed_config == "ds_config3.json":
        estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)

    #precision_factor = 4
    #print("Current model size : {} MB\n".format(get_n_params(model) * precision_factor / 1024. / 1024.))

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    with open(args.deepspeed_config) as fin :
        config = json.load(fin)

    # Overrides configuration json file 
    config["train_batch_size"] = int(args.batch_size)
    
    try:
        if args.cpu_offload_optim == '0':
            del config["zero_optimization"]["offload_optimizer"]
    except:
        pass

    try:
        if args.cpu_offload_param == '0':
            del config["zero_optimization"]["offload_param"]
    except:
        pass

    try:
        if args.fp16 == True:
            print("fp16 is true")
            config["zero_optimization"]["fp16"]["enabled"]  =True
            config["fp16"]["enabled"]                       =True
        elif args.fp16 == False:
            print("fp16 is false")
            config["zero_optimization"]["fp16"]["enabled"]  =False
            config["fp16"]["enabled"]                       =False
    except:
        pass


    model, optimizer, _, _ = deepspeed.initialize(
        args=args,
        config=config,
        model=model,
        #optimizer=optimizer,
        #model_parameters=optimizer_grouped_parameters,
        model_parameters=model.parameters(),
        dist_init_required=True)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.resume:
        # Use a local scope to avoid dangling references
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_sampler = None
    val_sampler = None

    # log csv filename
    if args.distributed :

        #prefix="logs/phase1_{}nodes_{}gpus_batch{}/".format(os.environ["GROUP_WORLD_SIZE"], os.environ["LOCAL_WORLD_SIZE"], args.batch_size)
        #filename_train ="{}/phase1_deepspeed_train_{}nodes_{}gpus_batch{}.csv".format(prefix, os.environ["GROUP_WORLD_SIZE"], os.environ["LOCAL_WORLD_SIZE"], args.batch_size)
        #filename_val   ="{}/phase1_deepspeed_val_{}nodes_{}gpus_batch{}.csv".format(prefix, os.environ["GROUP_WORLD_SIZE"], os.environ["LOCAL_WORLD_SIZE"], args.batch_size)

        prefix="logs/batch{}/".format(args.batch_size)
        filename_train ="{}/deepspeed_train_1nodes_1gpus_batch{}{}.csv".format(prefix, args.batch_size, args.suffix)
        filename_val   ="{}/deepspeed_val_1nodes_1gpus_batch{}{}.csv".format(prefix, args.batch_size, args.suffix)

    else :

        prefix="logs/batch{}/".format(args.batch_size)
        filename_train ="{prefix}/deepspeed_train_1nodes_1gpus_batch{batch_size}_bptt{bptt}_emb{emsize}_layers{layers}_hidden{hidden}_heads{heads}_{config}_offloadP{cpu_offload_param}_offloadO{cpu_offload_optim}{suffix}.csv".format(
                                                        prefix=prefix, 
                                                        batch_size=args.batch_size,
                                                        bptt=args.bptt,
                                                        emsize=args.emsize,
                                                        layers=args.layers,
                                                        hidden=args.hidden,
                                                        heads=args.heads,
                                                        config=args.deepspeed_config.split(".")[0].split("_")[1],
                                                        cpu_offload_param=args.cpu_offload_param,
                                                        cpu_offload_optim=args.cpu_offload_optim,
                                                        suffix=args.suffix
                                                    )
        filename_val   ="{}/phase1_deepspeed_val_1nodes_1gpus_batch{}.csv".format(prefix, args.batch_size, args.deepspeed_config.split(".")[0])


    if not args.distributed or (torch.distributed.get_rank() == 0):
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        try:
            os.remove(filename_train)
        except:
            print("File {} not found".format(filename_train))

        try:
            os.remove(filename_val)
        except:
            print("File {} not found".format(filename_val))

    if not args.distributed or (torch.distributed.get_rank() == 0):
        start_time = time.time()

    #########################################################
    # Training iterations
    #########################################################

    train_iter = WikiText2(split='train')
    #train_iter, val_iter, test_iter = WikiText2(split='train')
    train_data  = batchify(data_process(train_iter), args.batch_size)
    #val_data    = batchify(data_process(val_iter)  , args.batch_size)
    #test_data   = batchify(data_process(test_iter) , args.batch_size)

    epoch_time = AverageMeter()

    for epoch in range(int(args.epochs)):

        print("\nRunning epoch {}/{}\n".format(epoch, int(args.epochs)))

        timestamp = time.time()

        if epoch == 0 :
            output = train(model=model, train_data=train_data, bptt=args.bptt, epoch=epoch)

        elif epoch == 1 :
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
                output = train(model=model, train_data=train_data, bptt=args.bptt, epoch=epoch)
        #output = [0,0,0,0,0,0,0]

        if epoch > 0:
            epoch_time.update((time.time() - timestamp), 1)

        if not args.no_write_results :
            if (not os.path.isfile(filename_train)) or (os.path.getsize(filename_train) == 0) :
                with open(filename_train, "w") as fout:
                    fout.write('epoch,time,prep,forward,backward,update,throughput,cpu_util,cpu_mem_util,gpu_mem_util\n')
                    fout.flush()

        if epoch == 1 :

            if not args.no_write_results :

                prof.export_chrome_trace("{}.json".format(filename_train.split(".")[0]))

                #gpu_mem_util_MB         = gpu_utilization()
                gpu_mem_util_MB         = torch.cuda.memory_allocated() / (1024**2)
                print(f"GPU memory occupied: {gpu_mem_util_MB} MB.")

                #print('{},{},{},{},{},{},{},{},{},{}\n'.format(
                #                                            epoch,
                #                                            epoch_time.avg,
                #                                            output[0],
                #                                            output[1],
                #                                            output[2],
                #                                            output[3],
                #                                            output[4],
                #                                            output[5],
                #                                            output[6],
                #                                            gpu_mem_util_MB
                #                                            ))
                with open(filename_train, "a") as fout:

                    fout.write('{},{},{},{},{},{},{},{},{},{}\n'.format(
                                                                epoch,
                                                                epoch_time.avg,
                                                                output[0],
                                                                output[1],
                                                                output[2],
                                                                output[3],
                                                                output[4],
                                                                output[5],
                                                                output[6],
                                                                gpu_mem_util_MB
                                                                ))
                    fout.flush()

            return

        timestamp = time.time()

if __name__=="__main__":
    main()
