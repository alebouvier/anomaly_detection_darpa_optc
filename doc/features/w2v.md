# features/w2v.py

## Purpose
This module trains and uses a Word2Vec model on normalized process path and command-line data. It is the baseline semantic embedding pipeline for the project, before any LLM-based or transformer-based variant was introduced.

## Responsibilities
- preprocess Windows-style paths and command lines into token sequences
- train a Word2Vec model on those sequences
- compute contextual embeddings for path and command strings using the trained model
- expose a reusable embedding function for graph feature generation

## Inputs
- a list of raw path/command strings collected from processed client data
- a dataset config with regex patterns for splitting paths and commands
- a training configuration controlling window size and vector length

## Outputs
- a trained Word2Vec model saved to feature_data/w2v_model.pt
- embeddings for individual process strings that can be used in graph features

## Key Components
### Functions / Classes
- MySentences:
  - Purpose: iterable object used to feed the Word2Vec training corpus.
  - Inputs: file list, sampling ratio, config.
  - Outputs: token sequences for training.

- preprocess_path / preprocess:
  - Purpose: normalize and split path or command-line strings into tokens.

- train_val:
  - Purpose: train a Word2Vec model on the sampled token corpus.

- eval_for_encoding:
  - Purpose: take a raw value and produce a vector embedding using the trained Word2Vec model.

- eval_unknown:
  - Purpose: fill unknown tokens using context-based approximation when a token is absent from the vocabulary.

- eval_function_coeff_path_2combine / eval_function_coeff_path_decrois_geo / eval_function_coeff_path_const:
  - Purpose: weighting strategies used when aggregating token embeddings into one sequence embedding.

- eval:
  - Purpose: main aggregation routine producing the final sequence representation from a list of tokens.

- main:
  - Purpose: train the Word2Vec model and save it to disk.

### Important Workflow
1. Process paths and command lines are preprocessed into token lists.
2. The corpus is sampled and fed to a Word2Vec model.
3. Sequence embedding is computed by averaging token-level representations with weighting rules.
4. The trained model is saved and later reused in features/features_w2v.py and features/extract_features.py.

## Dependencies
- gensim
- numpy
- tqdm
- utils.utils

## Notes
- This file is the main Word2Vec baseline for the project.
- It is still used by the W2V feature pipeline and by some scripts that rely on a lightweight semantic encoder without transformer models.
- The training logic is intentionally simpler than the BERT workflow, but it is stable and historically important in this codebase.
