# OneNet
We use a variant of OneNet which is a unified neural network that jointly performs domain, intent, and slot predictions to reduce error propagation and facilitate information sharing.
For further details, please refer to the [paper](https://arxiv.org/abs/1801.05149).

## Example usage
We based our implementation on the [AllenNLP library](https://github.com/allenai/allennlp). For an introduction to this library, you should check [these tutorials](https://allennlp.org/tutorials).

```bash
$ PYTHONPATH=../../../../.. python train.py configs/basic.jsonnet -s serialization_dir
$ PYTHONPATH=../../../../.. python evaluate.py serialization_dir/model.tar.gz {test_file} --cuda-device {CUDA_DEVICE}
```

If you want to perform end-to-end evaluation, you can include the trained model by adding the model path (serialization_dir/model.tar.gz) to your ConvLab spec file.

## Data
We use the multiwoz data (data/multiwoz/[train|val|test].json.zip).

## References
```
@inproceedings{kim2017onenet,
  title={Onenet: Joint domain, intent, slot prediction for spoken language understanding},
  author={Kim, Young-Bum and Lee, Sungjin and Stratos, Karl},
  booktitle={2017 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  pages={547--553},
  year={2017},
  organization={IEEE}
}
```
