# Source files for prediction project

Files:
 - regression_test.m: 
  A test of the models I use for the project using data from Wagner et al. (2015) (Link: https://peerj.com/articles/1459/).
  For a combination of Ridge and lasso regularization terms, I loop through a range of parameters governing the strength of the terms to find one that leads to the lowest Mean Squared Error (MSE)
 - fish_abundance.m:
  Using Ridge and lasso linear models to predict fish abundance. 
  Produces graph of comparison of Ridge and lasso regressions over a range of regularization strengths
  Also produces graph to compare the chosen linear model with a possible logistic regression model
 - fish_species_richness.m:
  Using Ridge and lasso linear models to predict fish species richness. 
  Produces graph of comparison of Ridge and lasso regressions over a range of regularization strengths
