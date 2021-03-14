# Semi-supervised Gaussian Mixture Model [MATLAB]

* Semi-supervised Gaussian Mixture Model, for density estimation and classification, implemented in MATLAB.
* The code reproduces the example from this [MSSP paper](https://www.sciencedirect.com/science/article/pii/S088832702030039X).

## Demo Script
The script (demo.m) illustrates the potential increase in classification performance through semi-supervised model updates. Outputs of the script are shown below (3% of the training data are labelled).

### Conventional supervised learning

(blue ellipse indicates the prior)

![](images/supervised.png?raw=true)

Leads to classification accuracy: 88%

### Semi-supervised learning

(blue ellipse indicates the prior)

![](images/semisupervised.png?raw=true)

Leads to classification accuracy: 94%

(accuracy increase: 6.36%)
