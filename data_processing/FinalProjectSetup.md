# Gregory LeMasurier - UnifiedIO for VIMA Tasks

## Environment Setup
1. Download VIMA (link I used is replaced by a huggingface link, which also has the zip folder): https://huggingface.co/datasets/VIMA/VIMA-Data/tree/main
2. Extract ONLY the simple_manipulation task folder. VIMA is huge, we only need these files
3. Split into train, val, test by running:  
```bash
python data_processing/sort_data.py [/path/to/data]
```
4. Clean and create a directory with the minimal information needed by running:  
```bash
python data_processing/cleanDataset.py --path=[/path/to/data] --new_path=[OPTIONAL /path/to/put/clean/data OR LEAVE OUT TO NOT CREATE A NEW DIRECTORY]
```

install cuda (above 11.X above 11.1 NOT 12)
install cudnn for cuda version

conda install cuda -c nvidia
pip install --upgrade pip
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

in unifiedio repo:
pip install -r requirements.txt
pip install -e .