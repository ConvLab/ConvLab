# Sequicity

   Source code for the ACL 2018 paper entitled "Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence 
   Architectures" by Wenqiang Lei et al.

   ```
   @inproceedings{lei2018sequicity,
     title={Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures},
     author={Lei, Wenqiang and Jin, Xisen and Ren, Zhaochun and He, Xiangnan and Kan, Min-Yen and Yin, Dawei},
     year={2018},
     organization={ACL}
   }
   ```

   ## Training with default parameters

   ```
   python model.py -mode train -model [tsdf-camrest|tsdf-kvret]
   python model.py -mode adjust -model [tsdf-camrest|tsdf-kvret] -c lr=0.0003
   ```

   (optional: configuring hyperparameters with cmdline)

   ```
   python model.py -mode train -model [tsdf-camrest|tsdf-kvret] -c lr=0.003 batch_size=32
   ```

   ## Testing

   ```
   python model.py -mode test -model [tsdf-camrest|tsdf-kvret]
   ```

   ## Reinforcement fine-tuning

   ```
   python model.py -mode rl -model [tsdf-camrest|tsdf-kvret] -c lr=0.0001
   ```

   ## Before running

   1. Install required python packages. We used pytorch 0.3.0 and python 3.6 under Linux operating system. 

   ```
   pip install -r requirements.txt
   ```

   2. Make directories under PROJECT_ROOT.

   ```
   mkdir vocab
   mkdir log
   mkdir results
   mkdir models
   mkdir sheets
   ```

   3. Download pretrained Glove word vectors (glove.6B.50d.txt) and place them in PROJECT_ROOT/data/glove.

## What's new (Gaoxin)

Please download [data, trained model & results](https://drive.google.com/open?id=1R9VhYH4mbi5woqcmP422NjzN_IrGrKaF) and unzip the file here.

- Update the code for pytorch **1.0**
- Support the **multiwoz** dataset (see *data/MultiWoz* directory)
- Fill the placeholder slots by querying DBs
```
python model.py -mode train -model tsdf-multiwoz
python model.py -mode adjust -model tsdf-multiwoz -c lr=0.0003
python model.py -mode test -model tsdf-multiwoz
python model.py -mode rl -model tsdf-multiwoz -c lr=0.0001
```

- The model are saved at *models/multiwoz.pkl*, results are provided in the *results/multiwoz.csv*
- Interact with the agent using command line  (command END/RESET to end/reset the current dialog session)
```
python model.py -mode interact -model tsdf-multiwoz
```