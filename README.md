# contentHMM
## This repository contains code for paper: NextSum
*Note: The code is messy and only for documentational purpose. The author encourages the readers to only take this repository as a reference and implement their own version of models described in paper.*

### Data Preprocess
* Parse raw corpus: `preprocess.py`. Input data format should be NYT corpus raw file (.xml). Dependency: corenlpy
* Filter and group topics suitable for the experiment: `filter.py`

### Model
* Implementation of the Barzilay and Lee paper (Catching the drift: Probabilistic content models, with applications to generation and summarization): `content_hmm.py`. To train the HMM model, run `tagger_test.py`.
* Train a model to score word importance (the probability of the target word appearing in a summary): `attention_like_bilstm_wv.py`. Input data should be plain txt files. Then make predictions on NYT dataset:`pred_M.py`
* Generate encoding features of training data: First run `lexical_feat.py, interac_feat.py, incr_learn_feat.py`, then run `all_feat.py` to save encodings.
* Before training we compute all other features, then compile and shuffle the dataset: `rdn_sample_cand.py`
* Train the classification model for next sentence prediction: `ffnn_binclassif.py`

### Summary generation
* Beam search to generate summary: `generate.py`
* Code to evaluate generated summary: `eval_model.py`

### Others
* Baseline: `baseline.py`
* Experiments for ablation purpose (and using greedy search instead of beam search): `baselines.py`
