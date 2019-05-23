{
  "dataset_reader": {
    "type": "mle_policy",
    "num_actions": 1000
  },
  "train_data_path": "/home/sule/projects/research/DialogZone/data/multiwoz/train.json.zip",
  "validation_data_path": "/home/sule/projects/research/DialogZone/data/multiwoz/val.json.zip",
  "test_data_path": "/home/sule/projects/research/DialogZone/data/multiwoz/test.json.zip",
  // "train_data_path": "/home/sule/projects/research/DialogZone/data/multiwoz/sample.json",
  // "validation_data_path": "/home/sule/projects/research/DialogZone/data/multiwoz/sample.json",
  // "test_data_path": "/home/sule/projects/research/DialogZone/data/multiwoz/sample.json",
  "model": {
    "type": "vanilla_mle_policy",
    "input_dim": 192,
    "num_classes": 1000,
    "feedforward": {
      "input_dim": 192,
      "hidden_dims": 200,
      "num_layers": 1,
      "activations": "relu",
      "dropout": 0.5,
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 8 
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "validation_metric": "+accuracy",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 30,
    "grad_norm": 5.0,
    "patience": 10,
    "cuda_device": 0 
  },
  "evaluate_on_test": true
}
