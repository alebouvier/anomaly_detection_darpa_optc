# features/train_bert128.py

## Purpose
This module fine-tunes a compact BERT model on OpTC path and command-line text so that it can produce embeddings adapted to the process execution corpus. It is a lightweight language-model training step used to create a domain-specific encoder for downstream feature extraction.

## Responsibilities
- collect textual path/command samples from processed client data
- preprocess the textual content into token sequences
- train a masked-language-model BERT variant on those sequences
- save the fine-tuned tokenizer and model for later embedding generation

## Inputs
- list of client IDs
- dataset name
- a sampled fraction of content to use for training
- number of epochs and batch size
- dataset configuration containing regex patterns for path and command splitting

## Outputs
- a trained BERT model saved in the feature_data/Bert_ft directory
- a matching tokenizer saved in the same directory

## Key Components
### Functions / Classes
- PathCommandDataset:
  - Purpose: wraps a list of texts into a Hugging Face dataset with max-length tokenization.

- preprocess_path / preprocess:
  - Purpose: normalize Windows path and command-line strings into a token list.

- normalize_text_item:
  - Purpose: handle single strings or list-like values coming from processed data.

- build_text_sequence:
  - Purpose: convert a raw line into a clean text sequence for BERT training.

- get_sequence_embedding / encode_text:
  - Purpose: encode one text sample into a fixed-size vector at inference time.

- collect_texts:
  - Purpose: sample path/command lines from each client and build the training corpus.

- train_bert128:
  - Purpose: fine-tune a BERT model with masked language modeling.

- main:
  - Purpose: entry point to start the training process.

### Important Workflow
1. Raw path and command-line strings are sampled from each client’s processed files.
2. They are normalized and joined into natural-language-like sequences.
3. A BERT model is fine-tuned with masked-token prediction.
4. The trained model/tokenizer are saved to disk for later feature extraction.

## Dependencies
- torch
- transformers
- numpy
- utils.utils

## Notes
- This training step creates the domain-adapted encoder used by features/extract_features.py and features/features_bert.py.
- The resulting model is stored under feature_data/Bert_ft and is expected to be reused by the BERT-based feature generation workflow.
- This is a compact, focused encoder model rather than a full general-purpose language model.
