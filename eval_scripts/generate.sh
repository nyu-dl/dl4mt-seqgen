#!/bin/bash
# Various scripts for generating from models with different algorithms

best_deen_model="/misc/kcgscratch1/ChoGroup/mansimov/XLM-data/exp_elman/finetune_deen_tlm_uniform_8gpu_256batch_pickside_lr_debug/cjfn1o6tkv/best-valid_de-en_mt_bleu.pth"
best_ende_model="/misc/kcgscratch1/ChoGroup/mansimov/XLM-data/exp_elman/finetune_deen_tlm_uniform_8gpu_256batch_pickside_lr_evalcorr_debug/tv82a0o5es/best-valid_en-de_mt_bleu.pth"
gen_type=${2:-src2trg}
gpuid=${3:-0}
split=${4:-test}
n_iter=${5:-1}

if [ $gen_type = src2trg ];
  then
    model_path=$best_deen_model
  else
    model_path=$best_ende_model
fi

function left_right_greedy_1iter() {
  python ../adaptive_gibbs_sampler_simple.py --alpha 0.0 --beta 0.0 --gamma 1.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function left_right_beam_1iter() {
  python ../adaptive_gibbs_sampler_beam_simple.py --alpha 0.0 --beta 0.0 --gamma 1.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function uniform_greedy_1iter() {
  python ../adaptive_gibbs_sampler_simple.py --alpha 0.0 --beta 0.0 --gamma 0.0 --uniform --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function uniform_beam_1iter() {
  python ../adaptive_gibbs_sampler_beam_simple.py --alpha 0.0 --beta 0.0 --gamma 0.0 --uniform --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function least_most_greedy_1iter() {
  python ../adaptive_gibbs_sampler_simple.py --alpha 0.0 --beta 1.0 --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function least_most_beam_1iter() {
  python ../adaptive_gibbs_sampler_beam_simple.py --alpha 0.0 --beta 1.0 --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_greedy_1iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python ../adaptive_gibbs_sampler_simple.py --alpha 1.0 --beta ${beta} --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_beam_1iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python ../adaptive_gibbs_sampler_beam_simple.py --alpha 1.0 --beta ${beta} --gamma 0.0 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_notsimple_beam_1iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python -m pdb ../adaptive_gibbs_sampler_beam.py --alpha 1.0 --beta ${beta} --gamma 0.0 --use_data_length --num_topk_lengths 1 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function left_right_greedy_2iter() {
  python ../adaptive_gibbs_sampler_simple.py --alpha 0.0 --beta 0.0 --gamma 1.0 --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function left_right_greedy_4iter() {
  python ../adaptive_gibbs_sampler_simple.py --alpha 0.0 --beta 0.0 --gamma 1.0 --iter_mult 4 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function uniform_greedy_2iter() {
  python ../adaptive_gibbs_sampler_simple.py --alpha 0.0 --beta 0.0 --gamma 0.0 --uniform --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function least_most_greedy_2iter() {
  python ../adaptive_gibbs_sampler_simple.py --alpha 0.0 --beta 1.0 --gamma 0.0 --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_greedy_2iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python ../adaptive_gibbs_sampler_simple.py --alpha 1.0 --beta ${beta} --gamma 0.0 --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_greedy_4iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python ../adaptive_gibbs_sampler_simple.py --alpha 1.0 --beta ${beta} --gamma 0.0 --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function left_right_beam_2iter() {
  python ../adaptive_gibbs_sampler_beam_simple.py --alpha 0.0 --beta 0.0 --gamma 1.0 --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function uniform_beam_2iter() {
  python ../adaptive_gibbs_sampler_beam_simple.py --alpha 0.0 --beta 0.0 --gamma 0.0 --uniform --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function least_most_beam_2iter() {
  python ../adaptive_gibbs_sampler_beam_simple.py --alpha 0.0 --beta 1.0 --gamma 0.0 --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_beam_2iter() {
    if [ $gen_type = src2trg ];
      then
        beta=0.9
      else
        beta=1.0
    fi

  python ../adaptive_gibbs_sampler_beam_simple.py --alpha 1.0 --beta ${beta} --gamma 0.0 --iter_mult 2 --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function left_right_greedy_variter() {
  python ../adaptive_gibbs_sampler_simple.py --alpha 0.0 --beta 0.0 --gamma 1.0 --iter_mult ${n_iter} --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function uniform_greedy_variter() {
  python ../adaptive_gibbs_sampler_simple.py --alpha 0.0 --beta 0.0 --gamma 0.0 --uniform --iter_mult ${n_iter} --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function least_most_greedy_variter() {
  python ../adaptive_gibbs_sampler_simple.py --alpha 0.0 --beta 1.0 --gamma 0.0 --iter_mult ${n_iter} --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

function easy_first_greedy_variter() {
  python ../adaptive_gibbs_sampler_simple.py --alpha 1.0 --beta 0.9 --gamma 0.0 --iter_mult ${n_iter} --use_data_length --num_topk_lengths 4 --split $split --model_path ${model_path} --gen_type ${gen_type} --gpu_id ${gpuid}
}

if [ $1 == "gibbs" ]; then
    gibbs
elif  [ $1 == "left_right_greedy_1iter" ]; then
    left_right_greedy_1iter
elif  [ $1 == "left_right_beam_1iter" ]; then
    left_right_beam_1iter
elif  [ $1 == "uniform_greedy_1iter" ]; then
    uniform_greedy_1iter
elif  [ $1 == "uniform_beam_1iter" ]; then
    uniform_beam_1iter
elif  [ $1 == "least_most_greedy_1iter" ]; then
    least_most_greedy_1iter
elif  [ $1 == "least_most_beam_1iter" ]; then
    least_most_beam_1iter
elif  [ $1 == "easy_first_greedy_1iter" ]; then
    easy_first_greedy_1iter
elif  [ $1 == "easy_first_beam_1iter" ]; then
    easy_first_beam_1iter
  elif  [ $1 == "easy_first_notsimple_beam_1iter" ]; then
      easy_first_notsimple_beam_1iter
elif  [ $1 == "left_right_greedy_2iter" ]; then
    left_right_greedy_2iter
elif  [ $1 == "left_right_greedy_4iter" ]; then
    left_right_greedy_4iter
elif  [ $1 == "uniform_greedy_2iter" ]; then
    uniform_greedy_2iter
elif  [ $1 == "least_most_greedy_2iter" ]; then
    least_most_greedy_2iter
elif  [ $1 == "easy_first_greedy_2iter" ]; then
    easy_first_greedy_2iter
  elif  [ $1 == "easy_first_greedy_4iter" ]; then
      easy_first_greedy_4iter

elif  [ $1 == "left_right_beam_2iter" ]; then
    left_right_beam_2iter
elif  [ $1 == "uniform_beam_2iter" ]; then
    uniform_beam_2iter
elif  [ $1 == "least_most_beam_2iter" ]; then
    least_most_beam_2iter
elif  [ $1 == "easy_first_beam_2iter" ]; then
    easy_first_beam_2iter


elif  [ $1 == "left_right_greedy_variter" ]; then
    left_right_greedy_variter
elif  [ $1 == "uniform_greedy_variter" ]; then
    uniform_greedy_variter
elif  [ $1 == "least_most_greedy_variter" ]; then
    least_most_greedy_variter
elif  [ $1 == "easy_first_greedy_variter" ]; then
    easy_first_greedy_variter
fi
