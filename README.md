# moo-denovo

## Requirements and Installation

This package requires Anaconda Python 3.6 and the following packages:
* pyqt
* matplotlib
* pytorch
* rdkit
* pexpect
* tensorflow
* molsets (pip package)

`conda create --name moo-denovo python=3.6`
`conda activate moo-denovo`
`conda install pyqt matplotlib pytorch rdkit pexpect tensorflow`
`pip install molsets`

## Usage

Run the application with
`python app.py`

### Otimization
In the optimize tab you can select the number and the descriptors to optimize setting the miniumum and maximum value.

### Generation
In the generate tab you can use the optimized model to generate a molecular library 
