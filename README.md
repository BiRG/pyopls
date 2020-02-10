# pyopls
##### Orthogonal Projection to Latent Structures in Python. 

This package provides a scikit-learn-style regressor to perform OPLS(-DA). OPLS is a pre-processing method to remove
variation from the descriptor variables that are not correlated to the target variable (1).

The implementation provided here is equivalent to that of the [libPLS MATLAB library](http://libpls.net/).

A 1-component PLS regression is performed to evaluate the filtering.

## Notes
`pyopls` can only be used with a single target variable.
## Reference
1. Johan Trygg and Svante Wold. Orthogonal projections to latent structures (O-PLS).
   *J. Chemometrics* 2002; 16: 119-128. DOI: [10.1002/cem.695](https://dx.doi.org/10.1002/cem.695)
2. Eugene Edington and Patrick Onghena. "Calculating P-Values" in *Randomization tests*, 4th edition.
   New York: Chapman & Hall/CRC, 2007, pp. 33-53. DOI: [10.1201/9781420011814](https://doi.org/10.1201/9781420011814).
3. Johan A. Westerhuis, Ewoud J. J. van Velzen, Huub C. J. Hoefsloot, Age K. Smilde. Discriminant Q-squared for 
   improved discrimination in PLSDA models. *Metabolomics* 2008; 4: 293-296. 
   DOI: [10.1007/s11306-008-0126-2](https://doi.org/10.1007/s11306-008-0126-2)
