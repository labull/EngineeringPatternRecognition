# `EPR`

Code to reproduce paper results (or as close as possible, depending on 
data-availability). Each publication has a Jupyter notebook.

Mostly probabilistic/Bayesian ML for engineering applications, particularly 
performance and health monitoring. Scripts are provided to test and 
demonstrate the _EPR_ module.

## Notebooks for papers

* [Hierarchical Bayesian modelling for knowledge transfer across engineering fleets via multitask learning (CACAIE, 2022)](https://doi.org/10.1111/mice.12901)
  * Hierarchical regression models of engineering populations, allowing 
    knowledge transfer between subgroups.
  * Applications to truck fleet survival analysis and wind farm power 
    prediction.
  * Jupyter notebook [demo](https://github.com/labull/EngineeringPatternRecognition/tree/main/Knowledge-transfer-in-engineering-fleets.ipynb) 
    based on truck-fleet survival analysis.

* [On the transfer of damage detectors between structures: an experimental 
  case study (JSV, 2021)](https://doi.org/10.1016/j.jsv.2021.116072)
  * Domain adaptation to transfer novelty detectors between aircraft 
    tailplane ground-tests.
  * The [TCA code](https://github.com/labull/EngineeringPatternRecognition/tree/main/TCAdemo.py) used in the papers.

* [Towards semi-supervised and probabilistic classification in structural 
  health monitoring (MSSP, 2020)](https://doi.org/10.1016/j.ymssp.2020.106653)
  * Semi-supervised learning of GMMs via (MAP) expectation maximisation, 
    applied to Gnat aircraft ground-test data to utilise both labelled 
    and unlabelled data.
  * Jupyter notebook 
  [demo](https://github.com/labull/EngineeringPatternRecognition/tree/main/Semi-supervised-and-probabilistic-classification-in-SHM-MSSP2020.ipynb); 
  MATLAB [demo](https://github.com/labull/EngineeringPatternRecognition/tree/main/matlab/semi_supervised_GMM).

* [Probabilistic active learning: An online framework for structural health 
  monitoring (MSSP, 2019)](https://doi.org/10.1016/j.ymssp.2019.106294)
  * Uncertainty sampling (GMMs) to direct inspections from online data 
    streams from engineering systems. Applied to bridge, lathe, and
    aircraft monitoring datasets.
  * Jupyter notebook 
  [demo](https://github.com/labull/EngineeringPatternRecognition/tree/main/Probabilistic-active-learning-An-online-framework-for-SHM-MSSP2019.ipynb); 
  MATLAB [demo](https://github.com/labull/EngineeringPatternRecognition/tree/main/matlab/active_learning_GMM).

* [Active learning for semi-supervised structural health monitoring (JSV, 
  2018)](https://doi.org/10.1016/j.jsv.2018.08.040)
  * Hierarchical sampling for active learning (the DH active 
    learner) applied to learn a classifier for ground-test vibration data 
    from a Gnat aircraft.
  * MATLAB [demo](https://github.com/labull/EngineeringPatternRecognition/tree/main/matlab/cluster_based_active_learning).

## Algorithms

* Multitask Learning
    * [Hierarchical regression](https://www.taylorfrancis.com/books/mono/10.1201/9780429258411/bayesian-data-analysis-andrew-gelman-john-carlin-hal-stern-donald-rubin) (Stan)

* Domain Adaptation
  * [Transfer Component Analysis](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5640675&casa_token=Go7wJy20s-QAAAAA:-LZaw0y0LDV7TFK4ClCSoDbsPWF87A-GD2iklRy3ObjxL7A0lanOe92vM-UCd_WwJY7th6R3-SE) (TCA)

* Partially-supervised learning
  * [Active learning by uncertainty sampling in Gaussian Mixture Models](https://doi.org/10.1016/j.ymssp.2019.106294) 
    (GMMs)
  * [Semi-supervised learning of mixture models](https://www.morganclaypool.com/doi/pdfplus/10.2200/S00196ED1V01Y200906AIM006?casa_token=0YqCaqxyR1EAAAAA:v8kqB5LBhkclcS30fp0z9DOELXhwlPrqZV2YjJiAK2CuGAPNVoDgId_bODlX6mifibxb1ozTbio) 
  via (MAP) expectation 
    maximisation
  * [Hierarchical sampling for active learning](https://dl.acm.org/doi/pdf/10.1145/1390156.1390183?casa_token=MaX0vwAsl9kAAAAA:ADzBT6YbRvKUh6DfZOGB1O-eqO8q7v1JLTBLgcN263vjoROp4D6wc3MHkcwxMzX20cgPimPI-Ibx6g)
   (the DH active learner)

## Figures

### Multitask learning (MTL)

MTL for knowledge transfer between tasks

![](https://raw.githubusercontent.com/labull/EngineeringPatternRecognition/main/figures/corr-mdls.png)

Compared to independent models

![](https://raw.githubusercontent.com/labull/EngineeringPatternRecognition/main/figures/indep-mdls.png)

### Active learning

![](https://raw.githubusercontent.com/labull/EngineeringPatternRecognition/main/figures/uncertainty_sampling.png)

![](https://raw.githubusercontent.com/labull/EngineeringPatternRecognition/main/matlab/active_learning_GMM/images/38iisl.gif)

### Semi-supervised learning

![](https://raw.githubusercontent.com/labull/EngineeringPatternRecognition/main/figures/supervised_learning.png)

![](https://raw.githubusercontent.com/labull/EngineeringPatternRecognition/main/figures/semi-supervised_learning.png)

(blue ellipse shows the prior)

### TCA domain adaptation (transfer learning)

![](https://raw.githubusercontent.com/labull/EngineeringPatternRecognition/main/figures/TCAdemo_pca.png) ![](https://raw.githubusercontent.com/labull/EngineeringPatternRecognition/main/figures/TCAdemo_tca.png)

Archived MATLAB functions/scripts are available in the [matlab](https://github.com/labull/EngineeringPatternRecognition/tree/main/matlab) folder.
