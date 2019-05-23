{
  "dataset_reader": {
    "type": "mlst",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      },
    }
  },
  "train_data_path": "/home/sule/projects/research/DialogZone/data/multiwoz/train.json.zip",
  "validation_data_path": "/home/sule/projects/research/DialogZone/data/multiwoz/val.json.zip",
  "test_data_path": "/home/sule/projects/research/DialogZone/data/multiwoz/test.json.zip",
  // "train_data_path": "/home/sule/projects/research/DialogZone/data/multiwoz/sample.json",
  // "validation_data_path": "/home/sule/projects/research/DialogZone/data/multiwoz/sample.json",
  // "test_data_path": "/home/sule/projects/research/DialogZone/data/multiwoz/sample.json",
  "model": {
    "type": "mlst_nlu",
    "label_encoding": "BIO",
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 50,
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
            "trainable": true
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
            "embedding_dim": 16
            },
            "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": 128,
            "ngram_filter_sizes": [3],
            "conv_layer_activation": "relu"
            }
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 178,
      "hidden_size": 200,
      "num_layers": 2,
      "dropout": 0.5,
      "bidirectional": true
    },
    "feedforward": {
      "input_dim": 400,
      "hidden_dims": 200,
      "num_layers": 1,
      "activations": "linear",
      "dropout": 0.5,
    },
    "focal_loss_gamma": 0.5,
    // "focal_loss_gamma": 0,
    "regularizer": [
      [
        "scalar_parameters",
        {
          "type": "l2",
          "alpha": 0.1
        }
      ]
    ]
  },
  "iterator": {
    "type": "basic",
    "batch_size": 64
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "validation_metric": "+tag_f",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": 0
  },
  "evaluate_on_test": true
}
