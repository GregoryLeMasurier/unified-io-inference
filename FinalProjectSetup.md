# Gregory LeMasurier - UNIFIED-IO for VIMA Tasks

## Dataset Setup
1. Download VIMA (link I used is replaced by a huggingface link, which also has the zip folder): https://huggingface.co/datasets/VIMA/VIMA-Data/tree/main
2. Extract ONLY the simple_manipulation task folder. VIMA is huge, we only need these files
3. Split into train, val, test by running:  
```python data_processing/sort_data.py [/path/to/data]```
4. Clean and create a directory with the minimal information needed by running:  
```python data_processing/cleanDataset.py --path=[/path/to/data] --new_path=[OPTIONAL /path/to/put/clean/data OR LEAVE OUT TO NOT CREATE A NEW DIRECTORY]```

## Environment Setup
1. ```git clone https://github.com/GregoryLeMasurier/VIMA.git && cd VIMA```
2. ```pip install -e .```  
OR: ```pip install git+https://github.com/vimalabs/VIMA```
3. ```cd ..```
4. ```git clone https://github.com/GregoryLeMasurier/VIMABench.git && cd VimaBench```
5. ```pip install -e .```
6. ```cd ..```
7. ```git clone https://github.com/GregoryLeMasurier/unified-io-inference.git && cd unified-io-inference```

8. install cuda (above 11.X above 11.1 NOT 12)
9. install cudnn for cuda version

10. ```conda install cuda -c nvidia```
11. ```pip install --upgrade pip```
12. ```pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html```

13. ```pip install -r requirements.txt```
14. ```pip install -e .```
15. Download desired model (I used small and base)  
SMALL: ```wget https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/small_1000k.bin -O small.bin```  
BASE:  ```wget https://ai2-prior-uio.s3.us-west-2.amazonaws.com/public/model-weights-bin/base_1000k.bin -O base.bin```

## Run Training Loop
```python train.py --data_path=[/path/to/clean_data] --params_path=[/path/to/model] --model_size[small, base, large, xl] --batch_size=[OPTIONAL: batch_size] --learning_rate[OPTIONAL: learning rate] --epochs=[OPTIONAL: # epochs] --evaluate=[True,False] --checkpoint_path[OPTIONAL: /path/to/save] --enable_wandb=[OPTIONAL: True,False]```

## WANDB Link
https://wandb.ai/glemasurier/unified-io-inference?workspace=user-glemasurier

## Trained Models

## Report

## Presentation