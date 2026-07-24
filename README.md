# optc_ad_project

The project consists in detecting malicious events in the OpTC dataset leveraging link prediction methods for temporal graphs data.

This git repository contains a full pipeline from data prepocessing, link_prediction model training and anomaly detection.

## project origin and development

This project is based on the DyGLib library (a library implementing multiple temporal graph learning methods), a previous work from Florent Cheyron and the GRAAL project from Fanny Dijoub.

As many hypothesis and experiments have been tested during the implementation of the pipeline and as it has not been completely cleaned. Some portions of code or even entire scripts may be unused in the pipeline or obsolete.


## git organisation

Several branches are present in the git repository.
- main
- temporal_patterns: implement the insertion of test anomalies in the train data.
- model_diffusion: implement model diffusion for negative sampling generation
- visualisation: implement Umap visualisations (the visualisations are also implemented in the branch temporal_patterns)
- preproc_improvement: obsolete

## pipeline description

## examples of execution

Every task can be launched by executing the script optc_ad.py with different arguments.

* Transform raw log data into 3 csv files (egdes list, node features, edge features)
```shell
python src/optc_ad.py -t preprocessing
```

* Extract temporal pattern from the attack present in test data and incorporate them into train data 
```shell
python src/optc_ad.py -t temporal_pattern
```

* Fine-tuned a BERT model on textual node and edge features
```shell
python src/optc_ad.py -t train_bert128
```

* Create embedding from textual information
```shell
python src/optc_ad.py -t  feature_bert
```

* Train a link prediction model using inserted anomaly patterns as negative sampling.
```shell
python src/optc_ad.py -t train_link_prediction_ano_insertion --num_epochs 1 --negative_sample_strategy random
```

* Evaluate the model on validation and test data
```shell
python src/optc_ad.py -t validate_link_prediction_ano_insertion --num_epochs 1 --negative_sample_strategy random
```

* Compute metrics for the link prediction task and the anomaly detection task.
```shell
python src/optc_ad.py -t test_anomaly_detection
```

## Data organisation

the data folder contains all the necessary raw and intermediate data for teh executions. 

data/
├── feature_data
│   ├── bert128_training
│   ├── Bert_ft
│   ├── optc_051
│   └── optc_201
├── label_data
├── log_data
├── processed_data
│   ├── optc_051
│   └── optc_201
├── profiler_data
├── temporal_embeddings_data
│   └── optc_051
│   └── optc_201
├── test_result_data
│   ├── optc_051
│   └── optc_201
└── val_result_data
    ├── optc_051
    └── optc_201

* log_data: should contain the raw json/json.gz log data 
* label_data: should contain malicious.json (the malicious logs)
* processed_data: 
    * after the preprocessing task: contains the edge list, the raw node features and the raw edge features
    * after the feature_bert task: contains the embedded raw node and edge features
* feature_data: 
    * after train_w2v or train_bert128: contains the trained model in a pt file.
* val_result_data:
    * after validate_link_prediction_ano_insertion: contains 3 pkl files containing the predictions for the validation data
* test_result_data:
    * after validate_link_prediction_ano_insertion: contains 3 pkl files containing the predictions for the test data
* temporal_embeddings_data:
    * after train_link_prediction_ano_insertion: contains the temporal embeddings of the train data and of the negative sampling
    * after validate_link_predicition_ano_insertion: contains the temporal embeddings of the validation and test data and of the negative sampling.
* profiler_data:
    * after an execution: contains the execution profile (can be seen with snakeviz)



## experiments organisation

The experiments folder contains all the trained link prediction model, the monitoring information about the training (loss curve, ...), the metrics and plots for the link prediction and anomaly detection task.

experiments/
├── optc_051
│   ├── dygformer
│   ├── graphmixer
│   └── tgat
└── optc_201
    ├── dygformer
    ├── graphmixer
    └── tgat

experiments/optc_051/dygformer/
├── 2hop_neighbor_negative_sampling_DyGFormer_seed0
│   └── logs
├── ad_results
├── anomaly_detection
├── DyGFormer_seed0
│   ├── logs
│   ├── loss
│   └── saved_models
├── historical_negative_sampling_DyGFormer_seed0
│   └── logs
├── json
├── link_prediction
├── pdf_anom
├── popular_negative_sampling_DyGFormer_seed0
│   └── logs
├── random_negative_sampling_DyGFormer_seed0
│   └── logs
├── saved_results
├── visualisation
└── weird_predictions

* after train_link_ano_insertion
    * DyGFormer_seed0: 
        * logs: contains a log file with information on the training
        * loss: contains loss curve and csv file of the training
        * saved_models: contains the trained model

* after test_anomaly_detection
    * link_prediction:
        * a report file showing the metrics for the link prediction task (computed on validation set)
        * plots (ROC, AP curve, calibration)
    * anomaly_detection
        * a report file showing the metrics for the anomaly detection task at the edge level (computed on test set)
        * a report file showing the metrics for the anomaly detection task at the graph level (computed on test set)
        * plots at edge level (ROC, AP curve, calibration)


## Contact

For more information about the code, you can contact me at: alexandrebouvier6@gmail.com