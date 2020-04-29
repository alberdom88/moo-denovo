# moo-denovo

GUI of the method described in:
"De novo drug design of targeted chemical libraries based on artificial intelligence and multi-objective optimization" (submitted).
The code is build based on 
https://github.com/MarcusOlivecrona/REINVENT

## Requirements and Installation

This package requires Anaconda Python 3.6 and the following packages:
* pyqt
* matplotlib
* pytorch
* rdkit
* pexpect
* tensorflow
* molsets (pip package)

Installation:
* `conda create --name moo-denovo python=3.6`
* `conda activate moo-denovo`
* `conda install -c conda-forge rdkit`
* `conda install pyqt matplotlib pytorch pexpect tensorflow`
* `pip install molsets`

## Usage

Run the application with
`python app.py`

### Otimization
In the optimize tab you can select the number and the descriptors to optimize setting the miniumum and maximum value.
Opt field can be used, selecting the Similarity and UserFragment descriptor, in order to specify the SMILES of the reference molecule or fragment respectively. 

### Generation
In the generate tab you can use the optimized model to generate a molecular library.
First select the desired .ckpt file of the generative model obtained in the Optimization stage.
