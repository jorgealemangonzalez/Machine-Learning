
%% 1 Cargar imágenes

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
[imagesData, shapeData, labels, stringLabels] = extractData('../DB/CKDB', emotionsUsed);

%% 2 Convertir imágenes a formato Alexnet

% 636imgs x 128px x 128px => 227px x 227px x 3channel x 636imgs

imagesData = cat(4,imagesData,imagesData,imagesData);
imagesData = permute(imagesData, [2, 3, 4, 1]);
imagesData = imresize(imagesData, [227,227]);

%% 3 Crear división de train y test

K = 3;
indexes = crossvalind('Kfold',size(imagesData,4),K);

trainSamples = imagesData(:,:,:,indexes~=1);
trainLabels  = labels(indexes~=1);

testSamples  = imagesData(:,:,:,indexes==1);
testLabels   = labels(indexes==1);

%% 4 Aplicar CNN a las imagenes para obtener features

% Load pre-trained AlexNet
net = alexnet()
featureLayer = 'fc7';

% features = activations(net,X,layer,Name,Value)
% If X is an array of images, then the first three dimensions correspond 
% to height, width, and channels of an image (aka. 227x227x3xN), and the fourth dimension 
% corresponds to the image number.

trainFeatures = activations(net, trainSamples, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% 5 Entrenar SVM con trainFeatures

classifier = fitcecoc(trainFeatures, trainLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%% 6 Test SVM classifier

% Extract test features using the CNN
testFeatures = activations(net, testSamples, featureLayer, 'MiniBatchSize',32);

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))