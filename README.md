This repository is daption of LaserTagger for QDMR purpose.
Much of the credit belong to their work.

# LaserTagger
LaserTagger is a text-editing model which predicts a sequence of token-level
edit operations to transform a source text into a target text. The model
currently supports four different edit operations:

1. *Keep* the token.
2. *Delete* the token.
3. *Add* a phrase before the token.
4. *Swap* the order of input sentences (if there are two of them).

Operation 3 can be combined with 1 and 2. Compared to sequence-to-sequence
models, LaserTagger is (1) less prone to hallucination, (2) more data efficient,
and (3) faster at inference time.

A detailed method description and evaluation can be found in our EMNLP'19 paper:
[https://arxiv.org/abs/1909.01187](https://arxiv.org/abs/1909.01187)

LaserTagger is built on Python 3, Tensorflow and
[BERT](https://github.com/google-research/bert). It works with CPU, GPU, and
Cloud TPU.

## Usage Instructions

You can run all of the steps with

```
cmd run_qdmr_experiment.bat
```

Each step should be runned manually, and the script is self-explantory.
