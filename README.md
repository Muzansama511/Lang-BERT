# Lang-BERT
This BERT-style transformer incorporates "DeepNet" principles to enable deep training with post-Norm. The model is trained on OpenWebText (hf/Skylion007) and uses gpt-3/4 bit-level tokeniser using tiktoken. The model itself is built using the Lightning module to enable fast training and hardware modularity.
Dataset/data.py:get_dataloader requires a path to the OpenWebText database (get from hf/Skylion007). 

After HF download, process all the parquet files to train.bin and val.bin files using "Dataset/data.py:make_dataset_to_bin" function. 
Once the bin files are generated, one can start training directly with "python train.py".

Problems faced with environment.yaml: Need to download the appropriate version of torch (2.12) with necessary cuda support separately from https://pytorch.org/get-started/locally/
