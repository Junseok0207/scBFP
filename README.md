# Single-cell RNA-seq data imputation using Bi-level Feature Propagation

<p align="center">
    <a href="https://pytorch.org/" alt="PyTorch">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>

The official source code for "Single-cell RNA-seq data imputation using Bi-level Feature Propagation".

## Overview

<img width=100% src="Img/architecture.png"></img>

## Requirements
- Python version : 3.9.16
- Pytorch version : 1.10.0
- scanpy : 1.9.3

## Download data

Create the directory to save dataset.
```
mkdir dataset
```

You can download preprocessed data [here](https://www.dropbox.com/sh/eaujyhthxjs0d5g/AADzvVv-h2yYWaoOfs1sybKea?dl=0)

## How to Run

You can simply reproduce the result with following codes  
```
git clone https://github.com/Junseok0207/scBFP.git
cd scBFP
sh run.sh
```

## Hyperparameters

`--name:`
Name of the dataset.  
usage example :`--dataset baron_mouse`

`--gene_k:`
Number of neighbors in gene-gene graph  
usage example :`--k 10`

`--cell_k:`
Number of neighbors in cell-cell graph  
usage example :`--k 10`

`--gene_iter:`
Number of iterations in feature propagation using gene-gene graph  
usage example :`--iter 10`

`--cell_iter:`
Number of iterations in feature propagation using cell-cell graph  
usage example :`--iter 40`

Using above hyper-parmeters, you can run our model with following codes  

```
python main.py --name baron_mouse --gene_k 10 --cell_k 10 --gene_iter 10 --cell_iter 40
```

