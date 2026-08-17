# Lang-BERT
This BERT-style transformer incorporates "DeepNet" principles to enable deep training with post-Norm. The model is trained on OpenWebText (hf/Skylion007) and uses gpt-3/4 bit-level tokeniser using tiktoken. The model itself is built using the Lightning module to enable fast training and hardware modularity.
Dataset/data.py:get_dataloader requires a path to the OpenWebText database (get from hf/Skylion007). See [how to download the OpenWebText DB](https://github.com/Muzansama511/OpenWebText_setup).

Once the bin files are generated, one can start training directly with "python train.py".
