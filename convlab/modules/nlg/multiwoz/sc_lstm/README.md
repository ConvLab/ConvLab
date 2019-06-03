# nlg-sclstm-multiwoz
pytorch implementation of semantically-conditioned LSTM on multiwoz data


semantically-conditioned LSTM: https://arxiv.org/pdf/1508.01745.pdf

# Run the code

unzip [rar](https://drive.google.com/open?id=14EP8X-bcGgZqbOxQ_k2RSw_iJAMZvFiR) here



l=1

lr=0.005

model_path=./sclstm.pt

log=./sclstm.log

res=./sclstm.res



TRAIN
```
python3 run_woz.py  --mode=train --model_path=$model_path --n_layer=$l --lr=$lr > $log
```

TEST

```

python3 run_woz.py --mode=test --model_path=$model_path --n_layer=$l --beam_size=10 > $res

```

Calculate BLEU

```

python3 bleu.py --res_file=$res

```