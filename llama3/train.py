from model import LLAMA_MINI, Transformer

import os
import argparse

import deepspeed
from deepspeed.pipe import PipelineModule

import torch
from torch.utils.data import Dataset


class DummyLlamaDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size,(self.seq_len,)).long()
        labels = torch.randn(self.seq_len,self.vocab_size)
        return input_ids, labels


def get_args():
    parser = argparse.ArgumentParser(description='LLAMA')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def join_layers(model):
    layers = [
        model.tok_embeddings.to("cuda:0"),
        *[layer.to("cuda:0") for layer in model.layers],
        model.norm.to("cuda:0"),
        model.output.to("cuda:0")
    ]
    return layers

def train_pipe(args, part='parameters'):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    seqlen = 32

    net = Transformer(LLAMA_MINI, seqlen)
    net = PipelineModule(layers=join_layers(net),
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method=part,
                         activation_checkpoint_interval=0)

    trainset = DummyLlamaDataset(128, seqlen, LLAMA_MINI.vocab_size)

    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset)

    for step in range(args.steps):
        loss = engine.train_batch()

if __name__ == '__main__':
    args = get_args()

    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)

    train_pipe(args)