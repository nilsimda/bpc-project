# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neural machine translation (NMT) project for translating between Upper Sorbian (hsb) and German (de) using Facebook's Fairseq framework. The project uses a transformer architecture (IWSLT variant) trained on preprocessed parallel text data.

## Development Environment Setup

```bash
# Clone and setup environment
git clone git@github.com:nilsimda/bpc-project.git && cd bpc-project
uv venv --python 3.10  # Python 3.10 required for fairseq compatibility
uv pip install subword-nmt
uv pip install fairseq
source .venv/bin/activate
```

## Key Commands

### Data Preprocessing
```bash
./prepare.sh  # Full preprocessing pipeline: Moses → BPE → Fairseq binary
```

### Training
```bash
./train.sh  # Train transformer model with single GPU
```

### Evaluation
```bash
./eval.sh  # Generate translations using best checkpoint
```

## Architecture

### Data Processing Pipeline
1. **Moses preprocessing**: Normalize punctuation → Tokenize → Clean → Truecase
2. **Data splitting**: 80% train, 10% dev, 10% test
3. **BPE encoding**: 16K operations with min-frequency 2
4. **Fairseq preprocessing**: Convert to binary format for training

### Directory Structure
- `dataset/original/`: Raw parallel data (input.hsb, input.de)
- `dataset/output_moses/`: Moses preprocessing output and data splits
- `dataset/output_bpe/`: BPE codes and encoded files
- `dataset/fairseq/`: Binary dataset for fairseq training
- `checkpoints/`: Model checkpoints during training
- `moses_scripts/`: Perl scripts for text preprocessing

### Model Configuration
- Architecture: `transformer_iwslt_de_en`
- Training: 50 epochs max, early stopping (patience=10)
- Optimization: Adam with inverse sqrt LR schedule
- Evaluation: BLEU score with beam search (beam=5)
- GPU: Single GPU training/evaluation

## Important Notes

- Input files must be named `input.hsb` and `input.de` in `dataset/original/`
- BPE operations may need adjustment for very small datasets
- Model uses shared decoder input/output embeddings
- Checkpoints saved every epoch, best model selected by BLEU score
- Uses label smoothed cross-entropy loss with 0.1 smoothing