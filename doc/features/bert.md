# features/bert.py

## Purpose
This module builds text embeddings for process-related strings such as image paths and command lines using a BERT model. It is used as the low-level encoder that converts a normalized path or command string into a fixed-size vector used by higher-level feature builders.

## Responsibilities
- normalize and split Windows paths or command lines into token sequences
- convert those token sequences into BERT embeddings
- provide a reusable embedding function for the feature engineering pipeline

## Inputs
- a Hugging Face tokenizer and model
- a raw string containing either a path, parent path, or command line
- a configuration dictionary with model settings such as LEN_ENCODE_PATH
- a flag indicating whether the input is a command line or a path

## Outputs
- a NumPy vector representing the encoded text
- optionally a structured embedding using the same weighting logic used in the older GRAAL-inspired code

## Key Components
### Functions / Classes
- preprocess_path:
  - Purpose: normalizes path separators and tokenizes a file path into path elements.
  - Inputs: raw path string and configuration.
  - Outputs: list of tokens and whether the last item is an extension.

- preprocess_command_line:
  - Purpose: splits a command line into path and argument tokens while preserving structure.
  - Inputs: raw command line and configuration.
  - Outputs: list of sequence tokens.

- eval_for_encoding:
  - Purpose: main encoding entry point used by feature builders. It preprocesses the string, then calls the model to compute an embedding.
  - Inputs: tokenizer, model, data, is_command flag, and config.
  - Outputs: embedding vector.

- encode_text:
  - Purpose: produces a fixed-length embedding with padding/truncation to the configured vector size.
  - Inputs: tokenizer, model, text string, and config.
  - Outputs: numpy vector with length LEN_ENCODE_PATH.

### Important Workflow
1. The raw path or command line is normalized and split into meaningful tokens.
2. The token sequence is embedded by the BERT encoder.
3. The result is resized to a fixed dimension expected by the rest of the graph feature pipeline.

## Dependencies
- transformers
- torch
- utils.utils
- config values from the dataset configuration file

## Notes
- This file is adapted from the GRAAL project and contains additional helper functions that are not always used in the current pipeline.
- Some weighting functions such as eval_function_coeff_path_2combine and eval_function_coeff_path_decrois_geo are legacy helpers kept for compatibility with earlier experiments.
- In the current pipeline, the most relevant function is usually eval_for_encoding or encode_text.
