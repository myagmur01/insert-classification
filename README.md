
# Insert Classification
Predict if the given image has an insert or not. 

## Data
All rights reserved @[Overview.AI](https://www.overview.ai).

Data should be added to the `data/` directory, as in format of:

`train/good/good_*.png`

`train/bad/bad_*.png`

`validation/good/good_*.png`

`validation/bad/bad_*.png`

## Pre-requisites

The project was developed using python 3.7 with the following packages.
- Tensorflow 2.4
- Scikit-learn
- Opencv 4.1
- Flask


Installation with pip:

```bash
pip install -r requirements.txt
```

### Simple GUI
```bash
python app.py
```

### TODO: Run on Docker
```bash
docker build 
docker run 
```

### API Structure
- data/ : source data
- figures/ : data inspection figures
- models/ : trained model files
- results/ : training accuracy/loss/cf figures
- utils/ : training utility scripts  
- requirements.txt : pre-requiste libraries for the project
- approaches.txt : other strategies explained
- config.py : training and evaluation configurations
- insertnet.py: CNN models (simple_model and model_from_scratch)
- train.py : training script
- predict.py : model predictions
- app.py : Flask app

### Training and Evaluation:
Train:
```bash
python train.py --approach finetune
```
Predict:
```bash
python predict.py --model_path="models/.."
```

## Summary
There are numerous approaches to be followed, as summarized in approaches.txt. 
Only finetuning, weight balancing and focal loss were applied in this work so as to deal with imbalanced data problem

Thank you



