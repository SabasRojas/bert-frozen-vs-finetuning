# BERT Frozen vs Fine-Tuning

**Authors:** Sabas Rojas and Daniel Duru

## Introduction

This repository is our class re-implementation project based on BERT for text classification.

The full goal is to compare:

- Frozen BERT (feature extraction)
- Full fine-tuning

Paper used: Devlin et al., 2019, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.

Main contribution of the paper: strong language understanding results from bidirectional Transformer pre-training.

## Chosen Result

The key result we want to reproduce is the difference between frozen BERT and full fine-tuning on a text classification task.

It matters because it shows the compute/performance trade-off in transfer learning.

## Repo Contents

Project structure:

- `code/`: model setup, dataset pipeline, and training scripts
- `data/`: dataset usage notes/instructions
- `results/`: logs/checkpoints/metrics
- `poster/`: poster files (later)
- `report/`: report files (later)
- `README.md`: project description and reproduction guide
- `LICENSE`: MIT license

## Re-implementation Details

Current implementation (early phase):

- Backbone model: `bert-base-uncased`
- Dataset wired now: `ag_news`
- Tokenization: `AutoTokenizer` with truncation and configurable max length
- Training framework: PyTorch + Hugging Face Transformers
- Current runnable mode: frozen baseline only

Planned next additions:

- Full fine-tuning mode
- Evaluation script and metric comparison
- Additional dataset experiments

## Reproduction Steps

### Dependencies

- Python 3.9+ recommended
- Core libraries:
  - `torch`
  - `transformers`
  - `datasets`
  - `tqdm`

Install:

```bash
pip install -r code/requirements.txt
```

### Run Commands

Run current baseline:

```bash
python3 -m code.train --dataset ag_news --epochs 1 --batch_size 16
```

Useful command-line arguments:

- `--dataset {ag_news}`
- `--epochs`
- `--batch_size`
- `--learning_rate`
- `--max_length`
- `--seed`
- `--log_every_steps`

### Computational Resources

Minimum:

- CPU works (but slower).
- Around 8 GB RAM for small runs.

Recommended:

- CUDA-enabled GPU for faster training iterations.

## Current Results

Current repository state provides:

- Working frozen baseline training
- Printed training loss

What to expect after running now:

- Download of dataset/model on first run
- Loss logs per configured interval and average epoch loss
- No final benchmark comparison yet (pending)

## Conclusion

This repo currently contains the first part of the implementation (baseline setup + training). The next stages are fine-tuning, evaluation, and final analysis.

## References

1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT.
2. Hugging Face Transformers Documentation: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
3. Hugging Face Datasets Documentation: [https://huggingface.co/docs/datasets](https://huggingface.co/docs/datasets)
4. PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

## Acknowledgements

This work was done as part of course re-implementation coursework.
