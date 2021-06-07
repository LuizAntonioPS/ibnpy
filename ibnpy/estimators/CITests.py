# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:27:59 2021

@author: LuizPereira
"""
from scipy import stats
from pgmpy.estimators.CITests import g_sq
import numpy as np

def _chi_square_critical_value(p, dof): #deve estar aqui mesmo?
    """
    Calculate the critical value of the Chi-squared distribution. This method searchs
    the observation value for the provided probability that is less than or
    equal to the provided probability from the Chi-squared distribution.
    Parameters
    ----------
    p : float
        Probability //TO_DO.
    dof : int
        Degrees of freedom of the Chi-squared distribution.
    Returns
    -------
    Critical value: float
        The percentage point function at p on the Chi-Squared distribution with df degrees of freedom.
    Examples
    --------
    >>> from ibnpy.estimators import CITest
    >>> critical_value = CITests.chi_square_critical_value(0.99,10)
    >>> print(critical_value)
    23.209251158954356
    """
    return stats.chi2.ppf(p, dof);

def info_chi(X, Y, Z, data, boolean=True, significance=0.9):
    """
    Info-chi conditional independence test[1].
    Tests the null hypothesis that X is independent from Y given Zs.
    
    :math:`P(X,Y) = 2NI(X,Y)−χα,l`
    :math:`P(X,Y|Z) = 2NI(X,Y|Z)−χα,l`
    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set
    Y: int, string, hashable object
        A variable name contained in the data set, different from X
    Z: list, array-like
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []
    data: pandas.DataFrame
        The dataset on which to test the independence condition.
    significance: float # float == confidence_level in paper
        The `significance` is the probability (alpha value) used to calculate
        a critical value X_{α,l}
    boolean: bool
        If the function return is bool or not.
    Returns
    -------
    If boolean = False, returns: #olhar isto daqui e verificar se essas informações podem ser usadas e avaliar como a entrada dos dados pode ser afetada por isto.
        info_chi: float
            The info-chi test statistic.
    If boolean = True, returns:
        independent: boolean
            If the infochi_value of the test is equal to or lesser than 0,
            returns True. X and Y are independent
            Else returns False.
    References
    ----------
    [1] Shi, Da, and Shaohua Tan. "Incremental learning Bayesian network structures
        efficiently." 2010 11th International Conference on Control Automation
        Robotics & Vision. IEEE, 2010.
    [2] De Campos, Luis M., and Nir Friedman. "A scoring function for learning
        Bayesian networks based on mutual information and conditional independence
        tests." Journal of Machine Learning Research 7.10 (2006).
    Examples
    --------
    
    """
    if hasattr(Z, "__iter__"):
        Z = list(Z)
    else:
        raise (f"Z must be an iterable. Got object type: {type(Z)}")

    if (X in Z) or (Y in Z):
        raise ValueError(
            f"The variables X or Y can't be in Z. Found {X if X in Z else Y} in Z."
        )
        
    # Step 2: Calc 2NMI using G_test
    chi, p_value, dof = g_sq(X, Y, Z, data, boolean=False)
    
    # Step 3: Calc Chi Square Critical Value
    p = significance
    rx = data[X].nunique()
    ry = data[Y].nunique()
    rz = np.prod(data[Z].nunique()) #[2]
    dof = (rx-1)*(ry-1)*rz
    critical_value = _chi_square_critical_value(p, dof)
    
    # Step 4: Calc InfoChi
    infochi = chi - critical_value   

    if boolean:
        return infochi <= 0
    else:
        return infochi, chi, critical_value
    
    #return chi_square(X, Y, Z, data, boolean=boolean, significance_level=1-significance)
    
# %% testing
#import bnlearn
#values = bnlearn.sampling(bnlearn.import_DAG('alarm', CPD=True), n=1000)
#print(info_chi(X='FIO2', Y='PVSAT', Z=['SAO2'], boolean=False, data=values, significance=0.9))

# %% testing g-test // g-test = 2NMI
#from pgmpy.estimators.CITests import g_sq
#import bnlearn
#values = bnlearn.sampling(bnlearn.import_DAG('alarm', CPD=True), n=1000)
#g_sq(X='VENTTUBE', Y='VENTMACH', Z=['PAP'], data=values, boolean=False)