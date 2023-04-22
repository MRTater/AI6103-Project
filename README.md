# AI6103-Project: Diffusion
Repo for AI6103 project.

## Dataset
~~[Pokemon Dataset](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset?resource=download) by KVPRATAMA~~  
~~[Pokemon Dataset](https://www.kaggle.com/datasets/thedagger/pokemon-generation-one) by HARSHITDWIVEDI~~  
[Face Dataset](https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-1)

## DataLoader
```text
- AI6103-Project
    - data
    - train.py
    - test.py
    - dataloader.py
    - ...
```
### Data Acquisition
```shell
cd data
kaggle datasets download -d selfishgene/synthetic-faces-high-quality-sfhq-part-1
unzip selfishgene/synthetic-faces-high-quality-sfhq-part-1.zip
```
### Environment Setting
```shell
conda create -n diffusion python=3.9
conda activate diffusion
pip install -r requirements.txt
```
### Resize
```shell
python resize.py
```
## Training
```shell
python train.py --batch_size 128 --img_size 64 --epochs 500 --T 300 --dataset_folder "YourPathToTheDatasetFolder"
```
Only ```--dataset_folder``` is required. More parameter setting details can found in ```train.py```.

## Inferencing
```shell
python test.py --img_size 64 --model_path "YourPathToTheModel"
```
Only ```--model_path``` is required. More parameter setting details can found in ```test.py```.
