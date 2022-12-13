# Exploiting Global and Local Hierarchies for Hierarchical Text Classifications


## Preprocess

We follow the  repositories  of [contrastive-htc](https://github.com/wzh9969/contrastive-htc) and [HDLTex](https://github.com/kk7nc/HDLTex) to get the preprocessed datasets in json format file {'token': List[str], 'label': List[str]}.

Please download the origin datasets and pre-process them using the code in the corresponding folder:

+ [WoS](https://github.com/kk7nc/HDLTex) : `cd data/WebOfScience/ & python preprocess_wos.py`
+ [NYT](https://catalog.ldc.upenn.edu/LDC2008T19): `cd data/nyt/ &  python preprocess_nyt.py`
+ [RCV1-V2](https://github.com/ductri/reuters_loader): `cd data/rcv1/ & python preprocess_rcv1.py . & python data_rcv1.py`

## Train & Evaluation

``` shell
bash run_rcv1.sh

bash run_wos.sh

bash run_nyt.sh
```


Our Code is based on [s2s-ft](https://github.com/microsoft/unilm/tree/master/s2s-ft)
