# AttributeDisentangleToy
A toy experiment for attribute disentanglement.

## Instruction

1. Download the SUN Attribute Dataset.
```sh
wget https://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB.tar.gz 
wget https://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB_Images.tar.gz
```

2. Unzip the data and put them in the folder
```sh
../datasets/sun
```
relative to the code folder

3. To train the model (with weights), run
```sh
python train.py
```
and maybe after 10 epochs, stop the training, tune down the learing rate, and modify the epoch to start from 1 (0 is the pilot epoch) and continue training for another 10 epochs.

4. To see examples from the trained checkpoints, run
```sh
python draw_pic.py
```
Note that in the current code, it loads `adnet_corr.ckpt` (trained without re-weighing) and `adnet_weight3.ckpt` (trained with re-weighing).
