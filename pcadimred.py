import numpy as np
import matplotlib.pyplot as plt
#using variable org to define the data 
def PCAmodel(org):
    #1 mean center/normalize the data
    #We will do mean centering by subtracting mean from all features
    mean = np.mean(org, axis=0)
    print(mean.shape)
    mean_data = mean - org

    #2 compute the covariance matrix
    mat = np.cov(mean_data.T)
    mat = np.round(cov,2)
    print("covariance matrix", mat.shape)

    #3 Perform eigendecompositon on the covariance matrix
    #The eigenvector corresponding to the largest eigenvalue will give the
    #direction of max variance, which is the first principal component.
    eigenval, eigenvect = np.linalg.eig(mat)
    print(eigenvect)
    print(eigenval)

    #Sort eigenvects in descending order of their respective eigenvals
    range = np.arange(0, len(eigenval), 1)
    index = [x for _,x in sorted(zip(eigenval, range))][::-1]
    eigenval = eigenval[index]
    eigenvect = eigenvect[:,index]
    print(eigenvect)
    print(eigenval)


    #4Computing the explained variance and select components
    #We can select components by computing the explained variance of each 
    #feature
    sumeigenval = np.sum(eigenval)
    explained_var = eigenval/sumeigenval
    print(explained_var)
    cum_variance = np.cumsum((explained_var))
    print(cum_variance)

    #5 Now we will transform the data using eigenvectors, we will take the dot
    # product of these eigenvects with our data to get projections in the direction
    #of these eigenvects
    pca_trans = np.dot(mean_data, eigenvect)
    print(pca_trans.shape)

    #6 Plot the original data, mean centred data and transformed data in three plots