import os
import sys
import math
import errno
import argparse
import subprocess
import pickle as pkl
from tqdm import tqdm
import pdb
import ipdb

import numpy as np
import torch
import getpass
import operator

from src.data.loader import load_data
from src.data.dictionary import MASK_WORD
from src.utils import AttrDict, concat_batches, to_cuda, reload_checkpoint
from src.utils import create_batch, mask_batch_seq, create_masked_batch, generate_step, restore_segmentation
from src.trainer import SingleTrainer
from src.evaluation.evaluator import convert_to_text, eval_moses_bleu, SingleEvaluator

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def run(model, params, dico, data, split,
        src_lang, trg_lang,
        gen_type="src2trg", alpha=1., beta=1., gamma=0., uniform=False,
        iter_mult=1, mask_schedule="constant", constant_k=1,
        batch_size=8, gpu_id=0):
    #n_batches = math.ceil(len(srcs) / batch_size)
    if gen_type == "src2trg":
        ref_path = params.ref_paths[(src_lang, trg_lang, split)]
    elif gen_type == "trg2src":
        ref_path = params.ref_paths[(trg_lang, src_lang, split)]

    refs = [s.strip() for s in open(ref_path, encoding="utf-8").readlines()]
    hypothesis = []
    #hypothesis_selected_pos = []
    for batch_n, batch in enumerate(get_iterator(params, data, split, "de", "en")):
        (src_x, src_lens), (trg_x, trg_lens) = batch

        batches, batches_src_lens, batches_trg_lens, total_scores = [], [], [], []
        #batches_selected_pos = []
        for i_topk_length in range(params.num_topk_lengths):

            # overwrite source/target lengths according to dataset stats if necessary
            if params.de2en_lengths != None and params.en2de_lengths != None:
                src_lens_item = src_lens[0].item() - 2 # remove BOS, EOS
                trg_lens_item = trg_lens[0].item() - 2 # remove BOS, EOS
                if gen_type == "src2trg":
                    if len(params.de2en_lengths[src_lens_item].keys()) < i_topk_length + 1:
                        break
                    data_trg_lens = sorted(params.de2en_lengths[src_lens_item].items(), key=operator.itemgetter(1))
                    data_trg_lens_item = data_trg_lens[-1-i_topk_length][0] + 2
                    # overwrite trg_lens
                    trg_lens = torch.ones_like(trg_lens) * data_trg_lens_item
                elif gen_type == "trg2src":
                    if len(params.en2de_lengths[trg_lens_item].keys()) < i_topk_length + 1:
                        break
                    data_src_lens = sorted(params.en2de_lengths[trg_lens_item].items(), key=operator.itemgetter(1))
                    # take i_topk_length most likely length and add BOS, EOS
                    data_src_lens_item = data_src_lens[-1-i_topk_length][0] + 2
                    # overwrite src_lens
                    src_lens = torch.ones_like(src_lens) * data_src_lens_item

            if gen_type == "src2trg":
                sent1_input = src_x
                sent2_input = create_masked_batch(trg_lens, params, dico)
                dec_len = torch.max(trg_lens).item() - 2 # cut BOS, EOS
            elif gen_type == "trg2src":
                sent1_input = create_masked_batch(src_lens, params, dico)
                sent2_input = trg_x
                dec_len = torch.max(src_lens).item() - 2 # cut BOS, EOS

            batch, lengths, positions, langs = concat_batches(sent1_input, src_lens, params.lang2id[src_lang], \
                                                              sent2_input, trg_lens, params.lang2id[trg_lang], \
                                                              params.pad_index, params.eos_index, \
                                                              reset_positions=True,
                                                              assert_eos=True) # not sure about it

            if gpu_id >= 0:
                batch, lengths, positions, langs, src_lens, trg_lens = \
                    to_cuda(batch, lengths, positions, langs, src_lens, trg_lens)

            with torch.no_grad():
                batch, total_score_argmax_toks = \
                    _evaluate_batch(model, params, dico, batch,
                                    lengths, positions, langs, src_lens, trg_lens,
                                    gen_type, alpha, beta, gamma, uniform,
                                    dec_len, iter_mult, mask_schedule, constant_k)
            batches.append(batch.clone())
            batches_src_lens.append(src_lens.clone())
            batches_trg_lens.append(trg_lens.clone())
            total_scores.append(total_score_argmax_toks)
            #batches_selected_pos.append(selected_pos)

        best_score_idx = np.array(total_scores).argmax()
        batch, src_lens, trg_lens = batches[best_score_idx], batches_src_lens[best_score_idx], batches_trg_lens[best_score_idx]
        #selected_pos = batches_selected_pos[best_score_idx]

        #if gen_type == "src2trg":
        #    hypothesis_selected_pos.append([selected_pos, trg_lens.item()-2])
        #elif gen_type == "trg2src":
        #    hypothesis_selected_pos.append([selected_pos, src_lens.item()-2])

        for batch_idx in range(batch_size):
            src_len = src_lens[batch_idx].item()
            tgt_len = trg_lens[batch_idx].item()
            if gen_type == "src2trg":
                generated = batch[src_len:src_len + tgt_len, batch_idx]
            else:
                generated = batch[:src_len, batch_idx]
            # extra <eos>
            eos_pos = (generated == params.eos_index).nonzero()
            if eos_pos.shape[0] > 2:
                generated = generated[:(eos_pos[1,0].item()+1)]
            hypothesis.extend(convert_to_text(generated.unsqueeze(1), \
                                torch.Tensor([generated.shape[0]]).int(), \
                                dico, params))

        print("Ex {0}\nRef: {1}\nHyp: {2}\n".format(batch_n, refs[batch_n].encode("utf-8"), hypothesis[-1].encode("utf-8")))


    hyp_path = os.path.join(params.hyp_path, 'decoding.txt')
    hyp_path_tok = os.path.join(params.hyp_path, 'decoding.tok.txt')
    #hyp_selected_pos_path = os.path.join(params.hyp_path, "selected_pos.pkl")

    # export sentences to hypothesis file / restore BPE segmentation
    with open(hyp_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(hypothesis) + '\n')
    with open(hyp_path_tok, 'w', encoding='utf-8') as f:
        f.write('\n'.join(hypothesis) + '\n')
    #with open(hyp_selected_pos_path, 'wb') as f:
    #    pkl.dump(hypothesis_selected_pos, f)
    restore_segmentation(hyp_path)

    # evaluate BLEU score
    bleu = eval_moses_bleu(ref_path, hyp_path)
    print("BLEU %s-%s; %s %s : %f" % (src_lang, trg_lang, hyp_path, ref_path, bleu))
    # write BLEU score result to file
    result_path = os.path.join(params.hyp_path, "result.txt")
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("BLEU %s-%s; %s %s : %f\n" % (src_lang, trg_lang, hyp_path, ref_path, bleu))

def _evaluate_batch(model, params, dico, batch,
                    lengths, positions, langs, src_lens, trg_lens,
                    gen_type, alpha, beta, gamma, uniform,
                    dec_len, iter_mult, mask_schedule, constant_k):
    """Run on one example"""
    batch_size = batch.size(1)
    assert (batch_size == 1)
    assert gen_type in ["src2trg", "trg2src"]
    gen_pos = []

    iter_mult = min(dec_len, iter_mult)
    if mask_schedule == "constant": # HACK BECAUSE THE WAY THE CODE IS WRITTEN
        iter_mult += 1
    # log probabilities of previously present tokens at each position
    log_probs_tokens = torch.zeros(dec_len).cuda() # should be batch_size
    vocab_size = len(dico)
    mask_tok = dico.word2id[MASK_WORD]

    total_score_argmax_toks = 0
    not_chosen_pos = np.arange(dec_len)

    #selected_pos = []
    # do iter_mult passes over entire sentence
    # For constant-time MT decoding, don't multiply by dec_len
    for dec_iter in range(0, iter_mult):

        # select "best" position (topk)
        if mask_schedule == "linear":
            if dec_iter == 0:
                topk = dec_len
                max_poss = np.arange(topk)
            else:
                topk = max(int(dec_len * ((iter_mult - dec_iter) / iter_mult)), 1) # for short sequences
                if uniform:
                    max_poss = np.arange(topk)
                    np.random.shuffle(max_poss)
                    max_poss = max_poss[:topk]
                else:
                    if gamma == 1. and alpha == 0. and beta == 0.:
                        max_poss = np.arange(topk)
                    else:
                        _, max_poss = positions_prob.topk(topk, dim=-1)
        elif mask_schedule == "constant" and dec_iter > 0:
            topk = int(np.ceil(dec_len / iter_mult))
            if uniform:
                if len(not_chosen_pos) < topk:
                    not_chosen_pos = np.arange(dec_len)
                np.random.shuffle(not_chosen_pos)
                max_poss = not_chosen_pos[:topk]
                not_chosen_pos = np.delete(not_chosen_pos, np.arange(topk))
            else:
                if gamma == 1. and alpha == 0. and beta == 0.:
                    max_poss = np.arange(dec_len)[(dec_iter-1)*topk:dec_iter*topk]
                else:
                    _, max_poss = positions_prob.topk(topk, dim=-1)
        elif mask_schedule == "constant" and dec_iter == 0:
            max_poss = np.array([])
        elif mask_schedule == "all":
            topk = dec_len # predict for all positions
            max_poss = np.arange(topk)

        #print ("dec_iter {}, topk {}, dec_len {}".format(dec_iter, topk, dec_len))
        # create a prediction mask for positions to mask
        pred_mask = torch.zeros_like(batch).byte()
        if gen_type == "src2trg":
            pred_mask[src_lens[0]+1:-1, 0] = 1 # b_idx
        elif gen_type == "trg2src":
            pred_mask[1:src_lens[0]-1, 0] = 1 # b_idx

        if mask_schedule not in ["all"]:
            for pos in max_poss.tolist():
                if gen_type == "src2trg":
                    batch[src_lens[0] + pos + 1][0] = mask_tok # b_idx
                elif gen_type == "trg2src":
                    batch[pos + 1][0] = mask_tok # b_idx

        # re-run the model on the masked batch
        tensor = model('fwd', x=batch, lengths=lengths, positions=positions, langs=langs, causal=False)
        scores_pred = model('predict_wo_targets', tensor=tensor, pred_mask=pred_mask)

        # calculate log prob and prob for entropy
        log_probs_pred = torch.log_softmax(scores_pred, dim=1) # only compute softmax once
        probs_pred = torch.softmax(scores_pred, dim=1) # torch.exp(scores_pred)
        # calculate entropies and include normalization to put entropy and probability terms on same scale
        entropies = -(probs_pred * log_probs_pred).sum(dim=1)
        log_entropies = torch.log(entropies / torch.log(torch.tensor(vocab_size).float()))

        assert (log_entropies <= torch.log(torch.tensor(1.)).cuda()).all()

        # select "best" token (argmax) for ALL positions
        max_toks = scores_pred.argmax(dim=-1)

        # substitute predicted tokens in
        for pos in max_poss.tolist():
            tok = max_toks[pos]
            if gen_type == "src2trg":
                batch[src_lens[0] + pos + 1][0] = tok # b_idx
            elif gen_type == "trg2src":
                batch[pos + 1][0] = tok #b_idx
            total_score_argmax_toks += log_probs_pred[pos, tok].item() # b_idx

        if gen_type == "src2trg":
            curr_tokens = batch[src_lens[0]+1:-1]
        else:
            curr_tokens = batch[1:src_lens[0]-1]

        # gather the log probs for ALL current tokens
        if mask_schedule not in ["all"]:
            log_probs_tokens = torch.gather(log_probs_pred, 1, curr_tokens)[:, 0] # b_idx

        # get probability distribution over positions to choose from
        # we don't actually need to compute the softmax because it's monotonic
        #positions_prob = torch.softmax(-alpha * log_entropies - beta * log_probs_tokens + gamma * ltor_bias, dim=0)
        positions_prob = -alpha * log_entropies - beta * log_probs_tokens# + gamma * ltor_bias


    return batch, (total_score_argmax_toks/(iter_mult*dec_len))

def load_data_tmp(src_path, trg_path):
    """ Wrapper for loading aligned source and target sentences """

    def _load_file(path):
        with open(path, encoding="utf-8") as data_fh:
            data = data_fh.readlines()
        #data = [sent.replace("</s>", "") for sent in data]
        data = [('</s> %s </s>' % sent.strip()).split() for sent in data]
        return data

    src_sents = _load_file(src_path)
    trg_sents = _load_file(trg_path)

    assert len(src_sents) == len(trg_sents), "Found a different number of source and target sentences!"

    return src_sents, trg_sents


def get_iterator(params, data, data_set, lang1, lang2=None, stream=False):
    """
    Create a new iterator for a dataset.
    """
    assert data_set in ['valid', 'test']
    assert lang1 in params.langs
    assert lang2 is None or lang2 in params.langs
    assert stream is False or lang2 is None

    # hacks to reduce evaluation time when using many languages
    if len(params.langs) > 30:
        eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh", "ab", "ay", "bug", "ha", "ko", "ln", "min", "nds", "pap", "pt", "tg", "to", "udm", "uk", "zh_classical"])
        eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"])
        subsample = 10 if (data_set == 'test' or lang1 not in eval_lgs) else 5
        n_sentences = 600 if (data_set == 'test' or lang1 not in eval_lgs) else 1500
    elif len(params.langs) > 5:
        subsample = 10 if data_set == 'test' else 5
        n_sentences = 300 if data_set == 'test' else 1500
    else:
        # n_sentences = -1 if data_set == 'valid' else 100
        n_sentences = -1
        subsample = 1

    if lang2 is None:
        if stream:
            iterator = data['mono_stream'][lang1][data_set].get_iterator(shuffle=False, subsample=subsample)
        else:
            iterator = data['mono'][lang1][data_set].get_iterator(
                shuffle=False,
                group_by_size=True,
                n_sentences=n_sentences,
            )
    else:
        assert stream is False
        _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
        iterator = data['para'][(_lang1, _lang2)][data_set].get_iterator(
            shuffle=False,
            group_by_size=True, #False to preserve original split order
            n_sentences=n_sentences
        )

    for batch in iterator:
        yield batch if lang2 is None or lang1 < lang2 else batch[::-1]


def prepare_data(params, data, split, gen_type, alpha, beta, gamma, uniform, iter_mul, use_data_length, num_topk_lengths, mask_schedule, constant_k):
    """ Load data the same as in src/evaluation/evaluator.py """

    def create_reference_files():
        """
        Create reference files for BLEU evaluation.
        """
        params.ref_paths = {}

        for (lang1, lang2), v in data['para'].items():

            assert lang1 < lang2

            for data_set in ['valid', 'test']:

                # define data paths
                lang1_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_set))
                lang2_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_set))

                # store data paths
                params.ref_paths[(lang2, lang1, data_set)] = lang1_path
                params.ref_paths[(lang1, lang2, data_set)] = lang2_path

                # text sentences
                lang1_txt = []
                lang2_txt = []

                # convert to text
                for (sent1, len1), (sent2, len2) in get_iterator(params, data, data_set, lang1, lang2):
                    lang1_txt.extend(convert_to_text(sent1, len1, data['dico'], params))
                    lang2_txt.extend(convert_to_text(sent2, len2, data['dico'], params))

                # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
                lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

                # export hypothesis
                with open(lang1_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang1_txt) + '\n')
                with open(lang2_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lang2_txt) + '\n')

                # restore original segmentation
                restore_segmentation(lang1_path)
                restore_segmentation(lang2_path)

    params.hyp_path = os.path.join(params.dump_path, 'gen_fast/hypotheses_{0}_split_{1}_gentype_{2}_alpha{3}_beta{4}_gamma{5}_uniform{6}_itermul{7}_usedatalength{8}_numtopklengths{9}_masksched{10}_constantk{11}'.format(str(getpass.getuser()), split, gen_type, alpha, beta, gamma, uniform, iter_mul, use_data_length, num_topk_lengths, mask_schedule, constant_k))
    #params.hyp_path = os.path.join(params.dump_path, "debug")
    subprocess.Popen('mkdir -p %s' % params.hyp_path, shell=True).wait()
    create_reference_files()


def main(arguments):
    """ """
    parser = argparse.ArgumentParser(description='Enumerate over all possible positions to pick the best one')

    parser.add_argument('--model_path', type=str,
                        default='/misc/kcgscratch1/ChoGroup/mansimov/XLM-data/exp_elman/finetune_deen_tlm_uniform_4gpu_128batch_pickside_lr_debug/912lweev6s/best-valid_de-en_mt_bleu.pth',
                        help='path to pretrained TLM model')
    parser.add_argument('--src_lang', type=str, default='de', help='source language')
    parser.add_argument('--trg_lang', type=str, default='en', help='target language')
    parser.add_argument('--split', type=str, default='valid', help='use valid/test split of dataset', choices=['valid', 'test'])
    parser.add_argument('--use_data_length', action='store_true', help='use lengths according to dataset statistics')
    parser.add_argument('--num_topk_lengths', type=int, default=1, help='number of topk lengths to use when using dataset statistics')
    parser.add_argument('--beam_size', type=int, default=1, help='beam size to use in the experiments')
    parser.add_argument('--length_penalty', type=int, default=1, help='length penalty to use')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size to use')
    parser.add_argument('--gen_type', type=str, default="src2trg", \
                        choices=['src2trg', 'trg2src'], \
                        help='generation type to use src2trg (de->en) or trg2src (en->de)')
    parser.add_argument('--print_every', type=int, default=10, help='how often to log progress')
    parser.add_argument('--alpha', type=float, default=1., help='weight to put on entropy')
    parser.add_argument('--beta', type=float, default=1., help='weight to put on log prob')
    parser.add_argument('--gamma', type=float, default=0., help='weight to put on left to right decoding')
    parser.add_argument('--uniform', action='store_true', help='do uniform sampling of positions')
    parser.add_argument('--iter_mult', type=int, default=1, help='iteration multipler (multiply this number by target length)')
    parser.add_argument('--mask_schedule', type=str, choices=["constant", "linear", "all"],
                        default="linear", help='schedule for number of masks to predict at each iteration')
    parser.add_argument('--constant_k', type=int, default=1, help="If constant mask schedule, number of masks at each iteration")
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID, use -1 for CPU')
    args = parser.parse_args(arguments)

    if args.uniform:
        args.alpha, args.beta, args.gamma = 0, 0, 0

    # set GPU
    if args.gpu_id >= 0:
        torch.cuda.set_device(args.gpu_id)

    print("Evaluating model at {0}".format(args.model_path))
    # load everything from checkpoint
    params, dico, model = reload_checkpoint(args.model_path)
    # put on gpu
    model = model.cuda() if args.gpu_id >= 0 else model
    # put in eval model
    model = model.eval()

    if args.use_data_length:
        params.de2en_lengths = pkl.load(open(os.path.join(params.data_path, 'de2en_lengths.pkl'), 'rb'))
        params.en2de_lengths = pkl.load(open(os.path.join(params.data_path, 'en2de_lengths.pkl'), 'rb'))
        params.num_topk_lengths = args.num_topk_lengths
    else:
        params.de2en_lengths = None
        params.en2de_lengths = None
        params.num_topk_lengths = 1

    # load data
    params.eval_only = True
    params.batch_size = args.batch_size
    data = load_data(params)

    # creates reference files for BLEU eval
    prepare_data(params, data, args.split, args.gen_type,
                 args.alpha, args.beta, args.gamma, args.uniform,
                 args.iter_mult, args.use_data_length,
                 args.num_topk_lengths, args.mask_schedule, args.constant_k)

    # evaluate
    run(model, params, dico, data, args.split,
        args.src_lang, args.trg_lang, args.gen_type,
        args.alpha, args.beta, args.gamma, args.uniform,
        args.iter_mult, args.mask_schedule, args.constant_k,
        args.batch_size, args.gpu_id)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
