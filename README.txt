This repository contains the code of my final year project which proposes a 3D point cloud classification technique based on supervised machine learning techniques.
The model take both generic features such as XYZ point position, color, intensity and geometric features that derive from the study of the Principal Component Analysis
on those generic features.
The different codes had different functionality in the project:
> Pycloud contains the actual implementation of the neural network made by using the public machine learning model Random Forest, this code is used for training the neural network using the training dataset.
> ellipsoid_calculator takes in the dataset and calculate an ellipsoid of an arbitrary radius that cover at best the dimensionality of the set of points (basically defines if a set of point is mostly 1D, 2D or 3D)
> preprocessing calculates the geometric features of the dataset. it takes as input an ellipsoid calculating its PCA feature (Omnivariance, Planarity, Linearity)
> feature_extractor takes the gross dataset and extract the relevant features
> urban_classifier is the actual classificator and can be used once it has been trained from the training dataset
