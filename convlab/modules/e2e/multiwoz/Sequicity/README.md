# Sequicity

Sequicity is an end-to-end task-oriented dialog system based on a single sequence-to-sequence model that uses *belief span* to track dialog believes. We adapt the code from [github](https://github.com/WING-NUS/sequicity) to work in MultiWoz corpus.  The original paper can be found at [ACL Anthology](https://aclweb.org/anthology/papers/P/P18/P18-1133)

## Training with default parameters

   ```bash
$ PYTHONPATH=../../../../.. python model.py -mode train -model tsdf-multiwoz
$ PYTHONPATH=../../../../.. python model.py -mode adjust -model tsdf-multiwoz -c lr=0.0003
   ```

   (optional: configuring hyperparameters with cmdline)

   ## Testing

   ```bash
$ PYTHONPATH=../../../../.. python model.py -mode test -model tsdf-multiwoz
   ```

   ## Reinforcement fine-tuning

   ```bash
$ PYTHONPATH=../../../../.. python model.py -mode rl -model tsdf-multiwoz -c lr=0.0001
   ```

## What's new

Please download [data, trained model & results](https://drive.google.com/open?id=1rxeXFeCf30TlutvmPmkYponZQAuZrruI) and unzip the file here.

- Update the code for pytorch **1.0**
- Support the **multiwoz** dataset (see *data/MultiWoz* directory)
- Fill the placeholder slots by querying DBs
- The model are saved at *models/multiwoz.pkl*, results are provided in the *results/multiwoz.csv*
- Interact with the agent using command line  (command END/RESET to end/reset the current dialog session)

```bash
$ PYTHONPATH=../../../../.. python model.py -mode interact -model tsdf-multiwoz
```

## Data

We use the multiwoz data (./data/MultiWoz/[train|val|test].json)

## Reference

   ```
   @inproceedings{lei2018sequicity,
     title={Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures},
     author={Lei, Wenqiang and Jin, Xisen and Ren, Zhaochun and He, Xiangnan and Kan, Min-Yen and Yin, Dawei},
     year={2018},
     organization={ACL}
   }
   ```