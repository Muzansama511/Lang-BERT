# Lang-BERT
This BERT style transformer incorporates "DeepNet" principles to enable deep training with post-Norm. The model is trianed on OpenWebText (hf/Skylion007) and uses gpt-3/4 bit-level tokenizer using tiktoken. The model itself is buit using Lightning module to enable fast training and modularity in the hardware.
Dataset/data.py:get_dataloader requires a path to the OpenWebText database (get from hf/Skylion007). 

After hf download, process all the parquet files to train.bin and val.bin files using "Dataset/data.py:make_dataset_to_bin" funstion. 
Once the bin files are made, one can directly start training with "python train.py".

Problems faced with environment.yaml: Need to download the appropriate version of torch (2.13) with necessary cuda support seperately from https://pytorch.org/get-started/locally/
