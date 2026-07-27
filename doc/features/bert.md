# features/bert.py

## Purpose
evaluate a bert model on paths and command lines to get a representation. 

## Responsibilities
- preprocess the paths and command lines (split on "/", ...) to get a sequence of words
- create an embedding from this sequence.


## Key Components
### Functions / Classes
- eval_for_encoding:
  - Purpose: function used in feature_bert.py to preprocess and evaluate a bert model on paths and command lines. All other functions are called from this one except encode_text.
  - Inputs: 
    - the bert tokenizer
    - the bert model
    - the path or command line to evaluate
    - if it is a command line or a path
  - Outputs:
    - the embedding


## Dependencies
- utils.utils

## Notes
- This script has been adapted from the GRAAL project, so some functions are useless such as :
  - eval_function_coeff_path_2combine
  - eval_function_coeff_path_decrois_geo
