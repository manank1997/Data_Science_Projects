# Data_Science_Projects
A compilation presenting the implementation of various data science projects showcasing the knowledge of topics like clustering, classification, time-series analysis, network analysis, dimensionality reduction, phase space reconstruction, and recurrence plot analysis.

## Short Description of all files-
### Comparison_Regression_Techniques.ipynb
This project analyses the data of numerous Airbnb properties across the continental United States and tries to predict the price of these properties while considering many different criteria using various regression techniques. This project also compares the three regression techniques- Linear Regression, Random Forest Regression, and XGBoost Regression for the Airbnb dataset. The project also showcases some very cool visualizations highlighting pertinent dataset information.
### Correlation_Dimension.ipynb
The project discusses computes the intrinsic dimension of second order, ie, correlation dimension by first reproducing the phase space of a dynamical system. Then, calculate the distances between the data points and compute the false nearest neighbors for a given value of dimension. The algorithm continues increasing the dimension until no false nearest neighbors remain.
### Henons_Attractor.ipynb and Lorentz_Attractor.ipynb
These two projects show how we can generate these two dynamical systems using ordinary differential equation solvers and how we can perform phase space reconstruction.
### Image_Compression_PCA_SVD.ipynb
A very cool project that implements dimensionality reduction techniques for image compression. It is very useful for the encryption and compression of data. This project showcases how we can use Principal Component Analysis (PCA) and Singular Value Decomposition (SVD) to significantly reduce the dimensionality of an image while conserving the maximum amount of variance in our image's data.
### Introductory_RQA.ipynb
This project introduces the Recurrence Quantification Analysis (RQA) and shows its usage to predict the periodic or stochastic nature of the time series.
### Laplacian_Eigenmaps.ipynb
Another very cool project that shows how linear methods such as PCA fail when we have a non-linear dataset and how these methods incorrectly predict the dimensionality of a dataset. This project also showcases the implementation of a Machine Learning (ML) model that is non-linear and hence predicts the dimensions of a dataset accurately, even for non-linear datasets. We discuss two methods- Laplacian Eigenmaps and Isomaps. One of the highly important algorithms for data classification in the industry.
### community_detection.ipynb and Louvain_community_detection.ipynb
This project implements methods of network analysis while implementing network analysis tools (NetworkX library) and shows how we can deduce various communities in the dataset.
### Mutual_Information.ipynb
This project is a very useful implementation of big data with high level of complexity. In this project, we use the dataset of chicken pox cases in the country of Hungary between 2005 and 2015. We analyze the time series of number of cases in certain state of Hungary. We show how Pearson Correlation of multiple time series is not enough to explain the flow of information in the dataset, and how considering mutual information, or Shannon's entropy is a much more viable tool. We then compute the phase synchronization in these time series while applying Hilbert's transformations. We also show how the comparison of entropy value in a region affects the mutual information with other regions in the dataset.
### PCA.ipynb
An implementation of widely used data science technique called Principal Component Analysis (PCA) that shows how we can reduce the dimensionality of a dataset significantly and still be able to conserve the maximum amount of variance of the dataset. We use the penguins toy dataset and implement PCA to quantify that we can distribute the dataset based on three different species of penguins. It is a highly important clustering and classification algorithm for linear datasets.
### Phase_Space_Reconstruction.ipynb
In this project, we show how we can analyze a random time-series to reconstruct the phase space of an attractor that this time-series corresponds to. For this, we show how we can compute the embedding dimension using FNN (False Nearest Neighbors) method and time-delay using Auto-correlation technique for the time-series. And then we use matplotlib toolkits to reconstruct the phase space.
### RQA.ipynb and RQA_Practical_Example.ipynb
The first project explains computation of various RQA metrics and what they signify while analyzing the periodic, stochastic or real-life time series. The next project explains how to perform these computations in case of a real time-series. For this, we would need to compute the embedding dimension and time delay for the time series and consider the recurrence rate of 5% to retrieve the important information about the time series.
### Spectral_Clustering.ipynb
One of the most crucial project explaining how usage of k-means clustering is a bad idea and how it is inaccurate in most of the cases. This project also highlights a method of clustering called spectral clustering which is an upgrade to the k-means clustering and does an excellent job in recognising very complex clusters in the dataset.
### Intrinsic_Dimension.ipynb
This Project shows various methodologies of computing the intrinsic dimension of the dataset such as correlation dimension, K-nearest neighbors, and Information dimension.
