# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import sys
import pickle
import random
import inspect
import getpass
import argparse
import subprocess
import numpy as np
import torch
from torch import optim

from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

from .logger import create_logger

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

DUMP_PATH = '/checkpoint/%s/dumped' % getpass.getuser()
DYNAMIC_COEFF = ['lambda_clm', 'lambda_mlm', 'lambda_pc', 'lambda_ae', 'lambda_mt', 'lambda_bt']


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def reload_checkpoint(path):
    """ Reload params, dictionary, model from a given path """
    # Load dictionary/model/datasets first
    reloaded = torch.load(path)
    params = AttrDict(reloaded['params'])
    print("Supported languages: %s" % ", ".join(params.lang2id.keys()))

    # build dictionary / update parameters
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    params.n_words = len(dico)
    params.bos_index = dico.index(BOS_WORD)
    params.eos_index = dico.index(EOS_WORD)
    params.pad_index = dico.index(PAD_WORD)
    params.unk_index = dico.index(UNK_WORD)
    params.mask_index = dico.index(MASK_WORD)

    # build model / reload weights
    model = TransformerModel(params, dico, True, True)
    model.load_state_dict(reloaded['model'])

    return params, dico, model



def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    dump_path = DUMP_PATH if params.dump_path == '' else params.dump_path
    assert len(params.exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if params.exp_id == '':
        chronos_job_id = os.environ.get('CHRONOS_JOB_ID')
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        assert chronos_job_id is None or slurm_job_id is None
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            while True:
                exp_id = ''.join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()


class AdamInverseSqrtWithWarmup(optim.Adam):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
        decay_factor = lr * sqrt(warmup_updates)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        # linearly warmup for the first warmup_updates
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * warmup_updates ** 0.5
        for param_group in self.param_groups:
            param_group['num_updates'] = 0

    def get_lr_for_step(self, num_updates):
        # update learning rate
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.decay_factor * (num_updates ** -0.5)

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group['num_updates'] += 1
            param_group['lr'] = self.get_lr_for_step(param_group['num_updates'])


def get_optimizer(parameters, s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adam_inverse_sqrt':
        optim_fn = AdamInverseSqrtWithWarmup
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn(parameters, **optim_params)


def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    return [None if x is None else x.cuda() for x in args]


def restore_segmentation(path):
    """
    Take a file segmented with BPE and restore it to its original segmentation.
    """
    assert os.path.isfile(path)
    restore_cmd = "sed -i -r 's/(@@ )|(@@ ?$)//g' %s"
    subprocess.Popen(restore_cmd % path, shell=True).wait()


def parse_lambda_config(params):
    """
    Parse the configuration of lambda coefficient (for scheduling).
    x = "3"                  # lambda will be a constant equal to x
    x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease to 0 during the first 1000 iterations
    x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000 iterations, then will linearly increase to 1 until iteration 2000
    """
    for name in DYNAMIC_COEFF:
        x = getattr(params, name)
        split = x.split(',')
        if len(split) == 1:
            setattr(params, name, float(x))
            setattr(params, name + '_config', None)
        else:
            split = [s.split(':') for s in split]
            assert all(len(s) == 2 for s in split)
            assert all(k.isdigit() for k, _ in split)
            assert all(int(split[i][0]) < int(split[i + 1][0]) for i in range(len(split) - 1))
            setattr(params, name, float(split[0][1]))
            setattr(params, name + '_config', [(int(k), float(v)) for k, v in split])


def get_lambda_value(config, n_iter):
    """
    Compute a lambda value according to its schedule configuration.
    """
    ranges = [i for i in range(len(config) - 1) if config[i][0] <= n_iter < config[i + 1][0]]
    if len(ranges) == 0:
        assert n_iter >= config[-1][0]
        return config[-1][1]
    assert len(ranges) == 1
    i = ranges[0]
    x_a, y_a = config[i]
    x_b, y_b = config[i + 1]
    return y_a + (n_iter - x_a) * float(y_b - y_a) / float(x_b - x_a)


def update_lambdas(params, n_iter):
    """
    Update all lambda coefficients.
    """
    for name in DYNAMIC_COEFF:
        config = getattr(params, name + '_config')
        if config is not None:
            setattr(params, name, get_lambda_value(config, n_iter))


def set_sampling_probs(data, params):
    """
    Set the probability of sampling specific languages / language pairs during training.
    """
    coeff = params.lg_sampling_factor
    if coeff == -1:
        return
    assert coeff > 0

    # monolingual data
    params.mono_list = [k for k, v in data['mono_stream'].items() if 'train' in v]
    if len(params.mono_list) > 0:
        probs = np.array([1.0 * len(data['mono_stream'][lang]['train']) for lang in params.mono_list])
        probs /= probs.sum()
        probs = np.array([p ** coeff for p in probs])
        probs /= probs.sum()
        params.mono_probs = probs

    # parallel data
    params.para_list = [k for k, v in data['para'].items() if 'train' in v]
    if len(params.para_list) > 0:
        probs = np.array([1.0 * len(data['para'][(l1, l2)]['train']) for (l1, l2) in params.para_list])
        probs /= probs.sum()
        probs = np.array([p ** coeff for p in probs])
        probs /= probs.sum()
        params.para_probs = probs


def concat_batches(x1, len1, lang1_id, x2, len2, lang2_id, pad_idx, eos_idx, reset_positions, assert_eos=True):
    """
    Concat batches with different languages.
    """
    assert reset_positions is False or lang1_id != lang2_id
    lengths = len1 + len2
    if not reset_positions:
        lengths -= 1
    slen, bs = lengths.max().item(), lengths.size(0)

    x = x1.new(slen, bs).fill_(pad_idx)
    x[:len1.max().item()].copy_(x1)
    positions = torch.arange(slen)[:, None].repeat(1, bs).to(x1.device)
    langs = x1.new(slen, bs).fill_(lang1_id)

    for i in range(bs):
        l1 = len1[i] if reset_positions else len1[i] - 1
        x[l1:l1 + len2[i], i].copy_(x2[:len2[i], i])
        if reset_positions:
            positions[l1:, i] -= len1[i]
        langs[l1:, i] = lang2_id

    if assert_eos:
        assert (x == eos_idx).long().sum().item() == (4 if reset_positions else 3) * bs

    return x, lengths, positions, langs


def truncate(x, lengths, max_len, eos_index):
    """
    Truncate long sentences.
    """
    if lengths.max().item() > max_len:
        x = x[:max_len].clone()
        lengths = lengths.clone()
        for i in range(len(lengths)):
            if lengths[i] > max_len:
                lengths[i] = max_len
                x[max_len - 1, i] = eos_index
    return x, lengths

def create_batch(sentences, params, dico):
    """ Convert a list of tokenized sentences into a Pytorch batch

    args:
        sentences: list of sentences
        params: attribute params of the loaded model
        dico: dictionary

    returns:
        word_ids: indices of the tokens
        lengths: lengths of each sentence in the batch
    """
    bs = len(sentences)
    slen = max([len(sent) for sent in sentences])

    word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
    for i in range(len(sentences)):
        sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
        word_ids[:len(sent), i] = sent

    lengths = torch.LongTensor([len(sent) for sent in sentences])
    return word_ids, lengths

def create_masked_batch(lens, params, dico):
    """ Create a batch of all mask tokens of specified lengths.
    The first and

    args:
        lens (torch.Tensor): batch of sequences lengths of size (seq_len,)
        params: attribute params of the loaded model
        dico: dictionary
    returns:
        batch (torch.Tensor): batch of (seq_len, batch_size)
    """
    sents = []
    for _len in lens:
        sents.append([EOS_WORD] + ([MASK_WORD] * (_len.item() - 2)) + [EOS_WORD])
    return create_batch(sents, params, dico)[0]

def generate_step(logits, topk=1, temperature=1, return_list=True):
    """ Generate a word from from out[gen_idx]

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
    """
    if temperature is not None:
        logits /= temperature
    if isinstance(topk, str) and topk == "all":
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        kth_vals, kth_idx = logits.topk(topk, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    return idx.tolist() if return_list else idx

def mask_batch_seq(batch, src_lens, trg_lens, params, n_masks_per_step=1,
                  start_idxs=None, finished_gen=None, right2left=False, gen_type="src2tgt"):
    """ Create a prediction mask over a given batch
    by sampling for each target position,
    where the batch is concatenated source and target sentences
    args:
        batch (torch.Tensor):
        n_masks_per_step (int): number of elements to mask out
        start_idxs (int): if provided and there are no masks, indexes from which to start
            predicting n_preds_per_step consecutive tokens per example
            Assumes the indexes are in [0, {src/trg}_len] (i.e., don't add src len for trg)
        right2left (bool): if True, go right to left
    returns:
    """
    pred_mask = np.zeros(batch.size())
    mask_mask = (batch == params.mask_index)
    if mask_mask.is_cuda:
        mask_elts = mask_mask.nonzero().cpu().numpy()
        src_lens = src_lens.cpu()
        trg_lens = trg_lens.cpu()
    else:
        mask_elts = mask_mask.nonzero().numpy()

    for batch_idx, (src_len, trg_len) in enumerate(zip(src_lens, trg_lens)):

        if finished_gen[batch_idx]:
            continue

        # not clear about that part
        '''
        if mask_elts.size > 0 and mask_elts[np.where(mask_elts[:,1] == batch_idx)][:, 0].size > 0:
            row_masks = mask_elts[np.where(mask_elts[:,1] == batch_idx)][:, 0]
            start_idx = row_masks[-1] + 1 if right2left else row_masks[0]
        elif start_idxs is not None:
            start_idx = start_idxs[batch_idx].item()
            if gen_type == "src2tgt":
                start_idx += src_len
        else: # mask_elts is empty, so make row_masks empty too
            raise ValueError("No masks found and no starting index provided!")
        '''
        if start_idxs is not None:
            start_idx = start_idxs[batch_idx].item()
            if gen_type == "src2tgt":
                start_idx += src_len
        else: # mask_elts is empty, so make row_masks empty too
            raise ValueError("No masks found and no starting index provided!")

        assert 'start_idx' in locals(), pdb.set_trace() # hack for debugging, delete later

        if right2left: # right to left
            if gen_type == "src2tgt":
                end_idx = max(src_len, start_idx - n_masks_per_step)
            else:
                end_idx = max(0, start_idx - n_masks_per_step)
            pred_mask[end_idx:start_idx, batch_idx] = 1
        else: # left to right
            if gen_type == "src2tgt":
                end_idx = min(src_len + trg_len, start_idx + n_masks_per_step)
            else:
                end_idx = min(src_len, start_idx + n_masks_per_step)
            pred_mask[start_idx:end_idx, batch_idx] = 1

    #import pdb; pdb.set_trace()
    pred_mask = torch.from_numpy(pred_mask.astype(np.uint8))
    if mask_mask.is_cuda:
        pred_mask = pred_mask.cuda()
    pred_mask[batch == params.pad_index] = 0
    pred_mask[batch == params.eos_index] = 0  # TODO: remove
    pred_mask[batch == params.bos_index] = 0

    # targets
    targs = batch[pred_mask]
    # update input by filling with masks
    all_masks = targs.clone().fill_(params.mask_index)
    masked_batch = batch.masked_scatter(pred_mask, all_masks)
    return pred_mask, masked_batch, targs

def shuf_order(langs, params=None, n=5):
    """
    Randomize training order.
    """
    if len(langs) == 0:
        return []

    if params is None:
        return [langs[i] for i in np.random.permutation(len(langs))]

    # sample monolingual and parallel languages separately
    mono = [l1 for l1, l2 in langs if l2 is None]
    para = [(l1, l2) for l1, l2 in langs if l2 is not None]

    # uniform / weighted sampling
    if params.lg_sampling_factor == -1:
        p_mono = None
        p_para = None
    else:
        p_mono = np.array([params.mono_probs[params.mono_list.index(k)] for k in mono])
        p_para = np.array([params.para_probs[params.para_list.index(tuple(sorted(k)))] for k in para])
        p_mono = p_mono / p_mono.sum()
        p_para = p_para / p_para.sum()

    s_mono = [mono[i] for i in np.random.choice(len(mono), size=min(n, len(mono)), p=p_mono, replace=True)] if len(mono) > 0 else []
    s_para = [para[i] for i in np.random.choice(len(para), size=min(n, len(para)), p=p_para, replace=True)] if len(para) > 0 else []

    assert len(s_mono) + len(s_para) > 0
    return [(lang, None) for lang in s_mono] + s_para
