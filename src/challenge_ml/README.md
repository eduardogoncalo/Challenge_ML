<h1 align="center">Challenge ML</h1>

## Please if you'vent read  (README), please go back and follow the stpes to install and undertanding how Kedro Works

## Objectives


* Minimize the costs from a company with Maintance
* Build a scalable machine learning model with integrated and open tested pipelines using Kedro




## Data
 
The data is Split in Validation Dataset (2020) and Train Dataset (Pre 2020)·
All columns for bureaucratic reasons are coded.  For this reason any AED would’nt be so useful to  undertand the  problema.
Caracteristcis columns:

* 171 Columnas ( type: ‘Object’)
* 1 Feature Target ( Column: ‘class’)
*  170 Feature Training (insted missing value, has a string ‘na’ )



## Strategy 
This is a classic classification  problem, where the costs of pos-maintance are expensive then pre-mantaince where recall metric is the fit for the problem. So was create a cost function that need be minimize.
How our data is non-balance data (98% - negative - 2% - pos) i used undersample technique to keep the caracteristics of ‘pos’ samples.



<h1 align="center">Pipelines</h1>


* Feature Selection
* Classifier ML
* Validation Results



<img src="docs/images/kedroviz.png">
 
## Feature Selection

As coded columns and an analysis of them difficult to interpret, it was created a pipeline to find the best features that will be use the classifier_ml.
4 Steps to complet all the  imblearn.pipeline.

* https://imbalanced-learn.org/stable/references/pipeline.html#module-imblearn.pipeline

Preprocessing:
*    1- ReplaceStr - Replace ‘na’ to np.Nan (missing Values).  
*    2- CastFeature - Transform columns object to Float
*    3- Imputer - Replace missing values (np.Nan) by the medium.

UnderSample
* RandomUnderSampler -  How our data is non-balance data (98% - negative - 2% - pos), i used undersample tecicnique to mantein the caracteristics of ‘pos’ samples.

* https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html


RandomForest
SelectFeature



#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html