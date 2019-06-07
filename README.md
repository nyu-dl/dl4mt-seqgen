# A Generalized Framework of Sequence Generation with Application to Undirected Sequence Models

PyTorch implementation of the models described in the paper [A Generalized Framework of Sequence Generation with Application to Undirected Sequence Models](https://arxiv.org/abs/1905.12790).

The codebase is written on top of excellent implementation of cross-lingual masked language models from Facebook AI Research [https://github.com/facebookresearch/XLM](XLM)
Checkout that codebase for dependencies!

Download WMT'14 EN-DE [*valid/test* data](https://drive.google.com/file/d/149vO36tEE5Nyh7-pvqAs_3yhS7GfxME0/view?usp=sharing) and download [pretrained models](https://drive.google.com/open?id=1m1R7JC7tSnx3gog-UmeWamEERVoyWfMv).

To train the masked translation model for the purposes of generation run the following script in `train_scripts/train.sh`. Make sure to download the pretrained cross-lingual de-en masked language model from XLM repo.

To run the generation with different algorithms discussed in the paper checkout `eval_scripts/generate.sh` and `eval_scripts/generate-fast.sh`
