# optc_ad_project

The project consists in detecting malicious events in the OpTC dataset leveraging link prediction methods for temporal graphs data.

This git repository contains a full pipeline from data prepocessing, link_prediction model training and anomaly detection.

## project origin and development

This project is based on the DyGLib library (a library implementing multiple temporal graph learning methods), a previous work from Florent Cheyron and the GRAAL project from Fanny Dijoub.

As many hypothesis and experiments have been tested during the implementation of the pipeline and as it has not been completely cleaned. Some portions of code or even entire scripts may be unused in the pipeline or obsolete.


## git organisation

Several branches are present in the git repository.
- 

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
python src/optc_ad.py -t train_link_prediction_ano_insertion --num_epochs 5 --negative_sample_strategy random
```

* Evaluate the model on validation and test data
```shell
python src/optc_ad.py -t validate_link_prediction_ano_insertion --num_epochs 5 --negative_sample_strategy random
```

* Compute metrics for the link prediction task and the anomaly detection task.
```shell
python src/optc_ad.py -t test_anomaly_detection
```

## Data organisation

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


## experiments organisation

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


## Contact

For more information about the code, you can contact me at: alexandrebouvier6@gmail.com