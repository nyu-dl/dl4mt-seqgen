#!/bin/bash
# Various scripts for generating from models with different algorithms

best_deen_model="/misc/kcgscratch1/ChoGroup/mansimov/XLM-data/exp_elman/finetune_deen_tlm_uniform_8gpu_256batch_pickside_lr_debug/cjfn1o6tkv/best-valid_de-en_mt_bleu.pth"
best_ende_model="/misc/kcgscratch1/ChoGroup/mansimov/XLM-data/exp_elman/finetune_deen_tlm_uniform_8gpu_256batch_pickside_lr_evalcorr_debug/tv82a0o5es/best-valid_en-de_mt_bleu.pth"
gen_type=${2:-src2trg}
gpuid=${3:-0}
split=${4:-test}

if [ $gen_type = src2trg ];
  then
    model_path=$best_deen_model
  else
    model_path=$best_ende_model
fi

function uniform_lineardecay_20iter() {
  python ../adaptive_gibbs_sampler_fast.py --alpha 0.0 --beta 0.0 --gamma 0.0 --uniform --iter_mult 20 --mask_schedule linear --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function uniform_constant_20iter() {
  python ../adaptive_gibbs_sampler_fast.py --alpha 0.0 --beta 0.0 --gamma 0.0 --uniform --iter_mult 20 --mask_schedule constant --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function left_right_lineardecay_20iter() {
  python ../adaptive_gibbs_sampler_fast.py --alpha 0.0 --beta 0.0 --gamma 1.0 --iter_mult 20 --mask_schedule linear --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function left_right_constant_20iter() {
  python ../adaptive_gibbs_sampler_fast.py --alpha 0.0 --beta 0.0 --gamma 1.0 --iter_mult 20 --mask_schedule constant --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function least_most_lineardecay_20iter() {
  python ../adaptive_gibbs_sampler_fast.py --alpha 0.0 --beta 1.0 --gamma 0.0 --iter_mult 20 --mask_schedule linear --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function least_most_constant_20iter() {
  python ../adaptive_gibbs_sampler_fast.py --alpha 0.0 --beta 1.0 --gamma 0.0 --iter_mult 20 --mask_schedule constant --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function most_least_lineardecay_20iter() {
  python ../adaptive_gibbs_sampler_fast.py --alpha 0.0 --beta -1.0 --gamma 0.0 --iter_mult 20 --mask_schedule linear --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function most_least_constant_20iter() {
  python ../adaptive_gibbs_sampler_fast.py --alpha 0.0 --beta -1.0 --gamma 0.0 --iter_mult 20 --mask_schedule constant --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_lineardecay_20iter() {
  if [ $gen_type = src2trg ];
    then
      beta=0.9
    else
      beta=1.0
  fi
  python ../adaptive_gibbs_sampler_fast.py --alpha 1.0 --beta ${beta} --gamma 0.0 --iter_mult 20 --mask_schedule linear --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_constant_20iter() {
  if [ $gen_type = src2trg ];
    then
      beta=0.9
    else
      beta=1.0
  fi
  python ../adaptive_gibbs_sampler_fast.py --alpha 1.0 --beta ${beta} --gamma 0.0 --iter_mult 20 --mask_schedule constant --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function hard_first_lineardecay_20iter() {
  if [ $gen_type = src2trg ];
    then
      beta=0.9
    else
      beta=1.0
  fi
  python ../adaptive_gibbs_sampler_fast.py --alpha -1.0 --beta ${beta} --gamma 0.0 --iter_mult 20 --mask_schedule linear --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function hard_first_constant_20iter() {
  if [ $gen_type = src2trg ];
    then
      beta=0.9
    else
      beta=1.0
  fi
  python ../adaptive_gibbs_sampler_fast.py --alpha -1.0 --beta ${beta} --gamma 0.0 --iter_mult 20 --mask_schedule constant --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

if  [ $1 == "uniform_lineardecay_20iter" ]; then
    uniform_lineardecay_20iter
elif  [ $1 == "uniform_constant_20iter" ]; then
    uniform_constant_20iter
elif  [ $1 == "left_right_lineardecay_20iter" ]; then
    left_right_lineardecay_20iter
elif  [ $1 == "left_right_constant_20iter" ]; then
    left_right_constant_20iter
elif  [ $1 == "least_most_lineardecay_20iter" ]; then
    least_most_lineardecay_20iter
elif  [ $1 == "least_most_constant_20iter" ]; then
    least_most_constant_20iter
elif  [ $1 == "most_least_lineardecay_20iter" ]; then
    most_least_lineardecay_20iter
elif  [ $1 == "most_least_constant_20iter" ]; then
    most_least_constant_20iter
elif  [ $1 == "easy_first_lineardecay_20iter" ]; then
    easy_first_lineardecay_20iter
elif  [ $1 == "easy_first_constant_20iter" ]; then
    easy_first_constant_20iter
elif  [ $1 == "hard_first_lineardecay_20iter" ]; then
    hard_first_lineardecay_20iter
elif  [ $1 == "hard_first_constant_20iter" ]; then
    hard_first_constant_20iter
fi
