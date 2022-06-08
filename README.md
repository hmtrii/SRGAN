# image-super-resolution

## Dataset
### 1. Cyclegan dataset
- To download dataset, run:

    `./data/download_cyclegan_dataset.sh $subset`
- With `subset` is one of the following: *apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos.*

- The dataset is save under the structure:
```
data/
    cyclegan/
       subset_1/
            train/
                A/
                    *.jpg
                B/
                    *.jpg
            test/
                A/
                    *.jpg
                B/
                    *.jpg
        subset_2/
        ...
```
- Config the name of subset used for train/val/test in file `config.yml`
## Setup
```
conda create --name sr python=3.8.13 -y
conda activate sr
pip install -r requirements.txt
```
## Train
```
python train.py
```
- In default, the checkpoints and logging files are stored in `./runs/`
## Test
## Inference