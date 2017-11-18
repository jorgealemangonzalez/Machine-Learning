%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% P3 - RECONEIXEMENT DE PATRONS  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%    M�TODES DE CLASSIFICACI�    %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
addpath(genpath('.'));

%choose the emotion labels we want to classify in the database
% 0:Neutral 
% 1:Angry 
% 2:Bored 
% 3:Disgust 
% 4:Fear 
% 5:Happiness 
% 6:Sadness 
% 7:Surprise
emotionsUsed = [0 1 3 4 5 6 7];  

%%%%%%%%%%%%%%%% EXTRACT DATA %%%%%%%%%%%%
[imagesData, shapeData, labels, stringLabels] = extractData('../DB/CKDB', emotionsUsed);

%%%%%%%%%%%%%%%% EXTRACT FEATURES %%%%%%%%%%%%
grayscaleFeatures = extractFeaturesFromData(imagesData,'grayscale');


%%%%%%%%%%%%%%% DIVIDE DATA (TRAIN/TEST) WITH CROSS VALIDATION  %%%%%%%%%
K = 6;
indexesCrossVal = crossvalind('Kfold',size(imagesData,1),K);


%%%%%%%  EXAMPLE OF CLASSIFYING THE EXPRESSION USING TEMPLATE  MATCHING %%%%

% NO raw dada
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'Mahalanobis','PCA')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'Mahalanobis','LDA')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'Mahalanobis','PCAgaussianKernel')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'Mahalanobis','PCApolynomialKernel')
% 
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVM','none')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVM','PCA')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVM','LDA')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVM','PCAgaussianKernel')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVM','PCApolynomialKernel')

% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMpolynomialKernel','none')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMpolynomialKernel','PCA')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMpolynomialKernel','LDA')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMpolynomialKernel','PCAgaussianKernel')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMpolynomialKernel','PCApolynomialKernel')

% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMgaussian','none')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMgaussian','PCA')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMgaussian','LDA')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMgaussian','PCAgaussianKernel')
% [conf, acc] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMgaussian','PCApolynomialKernel')
