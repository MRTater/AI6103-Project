# AI6103-Project: Diffusion
Repo for AI6103 project. WIP.


## Dataset
~~[Pokemon Dataset](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset?resource=download) by KVPRATAMA~~  
~~[Pokemon Dataset](https://www.kaggle.com/datasets/thedagger/pokemon-generation-one) by HARSHITDWIVEDI~~
- [Face Dataset](https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-1)

## DataLoader
```text
- AI6103-Project
    - data
    - train.py
```
### Data Acquisition
```shell
cd data
kaggle datasets download -d selfishgene/synthetic-faces-high-quality-sfhq-part-1
unzip selfishgene/synthetic-faces-high-quality-sfhq-part-1.zip
```
### Resize
```shell
python resize.py
```
## Training
`python train.py`

## Testing
`python test.py`
