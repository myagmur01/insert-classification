
# Insert Classification
Predict if the given image has insert or not. 

## Data
All rights reserved @[Overview.AI](https://www.overview.ai).

Data should be added to the `data/` directory, as in format of:

`train/good/good_1.png`

`train/bad/bad_1.png`

`validation/good/good_*.png`

`validation/bad/bad_*.png`

## Pre-requisites

The project was developed using python 3.6.7 with the following packages.
- Numpy
- Scikit-learn
- imutils
- tensorflow
- opencv

Installation with pip:

```bash
pip install -r requirements.txt
```

### TODO: Simple GUI
Open the terminal in you machine and run the following command to access the web application in your localhost.
```bash
streamlit run app.py
```

### TODO: Run on Docker
Alternatively you can build the Docker container and access the application at `localhost:8051` on your browser.
```bash
docker build --tag app:1.0 .
docker run --publish 8051:8051 -it app:1.0
```
### API Structure
- data/ : source data
- figures/ : data inspection figures
- models/ : trained model files
- results/ : training accuracy/loss/cf figures
- requirements.txt : pre-requiste libraries for the project
- approaches.txt : other strategies explained
- config.py :
- insertnet.py:
- train.py :
- predict.py :
- app.py : Flask app

### Quick Run:
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
Only finetuning and weight balancing were applied in this work so as to deal with imbalanced data problem
Thank you

## Acknowledgements

[pyimagesearch](https://pyimagesearch.com/), for structurizing the classifier and implementing

[Tensorflow Guides](https://www.tensorflow.org/guide), for referencing built-in APIs.



