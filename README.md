# contentHMM
## This repository contains code for paper: NextSum
*Note: The code is messy and only for documentational purpose. The author encourages the readers to only take this repository as a reference and implement their own version of models described in paper.*

### Data Preprocess
* Parse raw corpus: `preprocess.py`. Input data format should be NYT corpus raw file (.xml)
* Filter and group topics suitable for the experiment: `filter.py`

### Model
* Implementation of the 2004 HMM paper: `content_hmm.py`. To train the HMM model, run `tagger_test.py`.
* Train a model to score importance given context: `attention_like_bilstm_wv.py`. Input data should be plain txt files. Then make prediction on NYT dataset:`pred_M.py`
* Generate encoding features of training data: First run `lexical_feat.py, interac_feat.py, incr_learn_feat.py`, then run `all_feat.py` to save encodings.
* Before training we compile and shuffle the dataset: `rdn_sample_cand.py`
* Train the classification model: `ffnn_binclassif.py`

### Summary generation
* Beam search to generate summary: `generate.py`
* Code to evaluate generated summary: `eval_model.py`

### Others
* Baseline: `baseline.py`
* Experiments for ablation purpose (and using greedy search instead of beam search): `baselines.py`
