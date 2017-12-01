function part2
% https://es.mathworks.com/help/nnet/examples/createsimple-deep-learning-network-for-classification.html
% https://es.mathworks.com/help/nnet/ref/activations.html

%Network architecture : http://cs231n.stanford.edu/reports/2016/pdfs/005_Report.pdf
dropout_probability = 0.5;
layers = [
    imageInputLayer([48 48 1]);
    convolution2dLayer([3 3],32);
    batchNormalizationLayer
    dropoutLayer(dropout_probability)
    reluLayer
    convolution2dLayer([3 3],64);
    maxPooling2dLayer(2)
    reluLayer
    
    fullyConnectedLayer(512);
    reluLayer
    
    fullyConnectedLayer(7, 'Name', 'PredictedLabel'); % output layer
    softmaxLayer
    classificationLayer('Name', 'ClassifiedLabels')
];

% DEEP NET BUT GRAT ACCURACY
% dropout_probability = 0.5;
% layers = [imageInputLayer([48 48 1]);
%           convolution2dLayer([3 3],64);         %FilterSize , NumFilters
%           batchNormalizationLayer
%           dropoutLayer(dropout_probability)
%           maxPooling2dLayer(2)
%           reluLayer
%           convolution2dLayer([5 5],128);         %FilterSize , NumFilters
%           batchNormalizationLayer
%           dropoutLayer(dropout_probability)
%           maxPooling2dLayer(2)
%           reluLayer
%           convolution2dLayer([3 3],512);         %FilterSize , NumFilters
%           batchNormalizationLayer
%           dropoutLayer(dropout_probability)
%           maxPooling2dLayer(2)
%           reluLayer
%           convolution2dLayer([3 3],512);         %FilterSize , NumFilters
%           batchNormalizationLayer
%           dropoutLayer(dropout_probability)
%           maxPooling2dLayer(2)
%           reluLayer
%           
%           fullyConnectedLayer(256);
%           batchNormalizationLayer
%           dropoutLayer(dropout_probability)
%           reluLayer
%           fullyConnectedLayer(512);
%           batchNormalizationLayer
%           dropoutLayer(dropout_probability)
%           reluLayer
%           
%           fullyConnectedLayer(7); % output layer
%           softmaxLayer
%           classificationLayer];

options = trainingOptions('sgdm','ExecutionEnvironment','cpu');
rng('default')

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

%emotionsUsed = [0 1 3 4 5 6 7];

rootFolder = '../CKDB';
categories = {'0','1','3','4','5','6','7'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

%% 1.2 Aislar el mismo numero de samples por label

tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});
imds = splitEachLabel(imds, minSetCount, 'randomize'); % Dejar 25 de cada
%% 2 Convertir imágenes a formato Alexnet

imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
    function Iout = readAndPreprocessImage(filename)
                
        I = imread(filename);
        
        % Resize the image as required for the CNN. 
        Iout = imresize(I, [48 48]);  
    end

%% 3 Crear división de train y test

[trainSamples, testSamples] = splitEachLabel(imds, 0.3, 'randomize');

%% Train net

% net = trainNetwork(trainSamples,layers,options);
load net
save('net.mat','net');
%predictedLabels = activations(net, testSamples, 'ClassifiedLabels', 'MinibatchSize', 30)
predictedLabels = classify(net, testSamples);

%% 6 Accuracy and conf matrix

% Tabulate the results using a confusion matrix.
testLabels = testSamples.Labels;
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))

% Compute accuracy
accuracy = sum(diag(confMat)) / sum(confMat(:))
end