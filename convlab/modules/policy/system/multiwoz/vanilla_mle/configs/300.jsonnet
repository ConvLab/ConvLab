{
  "dataset_reader": {
    "type": "mle_policy",
    "num_actions": 300
  },
  "train_data_path": "/home/sule/projects/research/cvdev/data/multiwoz/train.json.zip",
  "validation_data_path": "/home/sule/projects/research/cvdev/data/multiwoz/val.json.zip",
  "test_data_path": "/home/sule/projects/research/cvdev/data/multiwoz/test.json.zip",
  "model": {
    "type": "vanilla_mle_policy",
    "input_dim": 392,
    "num_classes": 300
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
    "validation_metric": "+accuracy",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 30,
    "grad_norm": 5.0,
    "patience": 10,
    "cuda_device": 0 
  },
  "evaluate_on_test": true
}
