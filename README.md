# BPFun
## Introduction
## Related Files
### BPFun
| File Name   | Description |
| ----------- | ----------- |
| getdata.py      | get data and process       |
| train.py   | train model        |
| test.py   | test model result        |
| predictor.py   | use model prediction        |
| data   | data        |
| test   | instance test data        |
| aaindex1_test.py   | aaindex1 encoded        |
| aaindex1.my.csv   | physicochemical properties of amino acids        |
| esm_test.py   | ESM-2 model        |
| prot_t5.py   | ProtT5 model        |
| transformer.py   | transformer encode and positional embedding         |
| evalution.py   | evaluation metrics (for evaluating prediction results)        |
| model.py   | model construction        |
## Requirements
python==3.7
Keras==2.10.0
Keras-Preprocessing==1.1.2
numpy==1.21.1
pandas==1.5.2
scikit-learn==1.3.0
scipy==1.10.1
tensorboard==2.10.1
tensorflow-gpu==2.10.1
## How to use
1.get data  (ProtT5 and ESM-2 need to download models)
    
    `python3 getdata.py`
2.train  

    `python3 train.py`
3.test  

    `python3 test.py`
4.predictor  

    `python3 predictor.py`
