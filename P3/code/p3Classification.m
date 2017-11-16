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
[imagesData shapeData labels stringLabels] = extractData('../CKDB', emotionsUsed);

%%%%%%%%%%%%%%%% EXTRACT FEATURES %%%%%%%%%%%%
grayscaleFeatures = extractFeaturesFromData(imagesData,'grayscale');


%%%%%%%%%%%%%%% DIVIDE DATA (TRAIN/TEST) WITH CROSS VALIDATION  %%%%%%%%%
K = 6;
indexesCrossVal = crossvalind('Kfold',size(imagesData,1),K);


%%%%%%%  EXAMPLE OF CLASSIFYING THE EXPRESSION USING TEMPLATE  MATCHING %%%%
%[CONF ACC] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'example','none')
%[CONF ACC] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVM','none')
[CONF ACC] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVM','PCA')
%[CONF ACC] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVM','LDA')
%[CONF ACC] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVM','kernelPCAgaussian')
%[CONF ACC] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVM','kernelPCApolynomial')
 
% [CONF1 ACC1] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMpolynomialKernel','none')
% [CONF2 ACC2] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMpolynomialKernel','PCA')
% [CONF3 ACC3] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMpolynomialKernel','LDA')

%[CONF1 ACC1] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMgaussian','none')
%[CONF2 ACC2] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMgaussian','PCA')
%[CONF3 ACC3] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'SVMgaussian','LDA')


%[CONF ACC] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, 'Mahalanobis','minPCA')
