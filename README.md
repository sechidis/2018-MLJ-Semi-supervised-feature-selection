# MLJ 2018 - Simple Strategies for Semi-Supervised Feature Selection

Matlab code for the methods presented in:

K. Sechidis, G. Brown, Simple Strategies for Semi-Supervised Feature Selection. <br /> https://link.springer.com/article/10.1007/s10994-017-5648-2

## Hypothesis testing in semi-supervised scenarios (Section 3)
Function semiIAMB.m implements our algorithm Semi-IAMB, which is the switching procedure applied to Markov Blanket discovery IAMB (IAMB.m).

## Ranking features in semi-supervised scenarios (Section 4)
Functions semiMIM.m and semiJMI.m implement our algorithms Semi-MIM and Semi-JMI, which are the switching procedure applied to the feature selection methods MIM (MIM.m) and JMI (JMI.m) respectively.

## Tutorial
The tutorial 'Tutorial_SemiSupervised_FS.m' presents how our suggested methods can be used for feature selection in semi-supervised learning environments.

## Citation

If you make use of the code found here, please cite the paper above.

@article{sechidis2017semisupervised,<br />
title = {Simple strategies for semi-supervised feature selection},<br />
author = {Konstantinos Sechidis and Gavin Brown},<br />
journal = {Machine Learning},<br />
volume = {107},<br />
number = {2},<br />
pages = {357--395},<br />
year = 2018<br />
} 
