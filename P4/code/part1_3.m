
function part1_3

rng('default');

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

rootFolder = '../DB/CKDB';
categories = {'0','1','3','4','5','6','7'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

% 1.2 Aislar el mismo numero de samples por label con Bagging
% => Repetir imagenes hasta obtener 4 veces el mínimo

tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});
imdsTemp = splitEachLabel(imds, minSetCount, 'randomize');
imds.Files = [imdsTemp.Files; imdsTemp.Files; imdsTemp.Files; imdsTemp.Files];
imds.Labels = [imdsTemp.Labels; imdsTemp.Labels; imdsTemp.Labels; imdsTemp.Labels];

%% 2 Convertir imágenes a formato Alexnet

imds.ReadFcn = @(filename)readAndPreprocessImage(filename);
    function Iout = readAndPreprocessImage(filename)
                
        I = imread(filename);
        
        % Some images may be grayscale. Replicate the image 3 times to
        % create an RGB image. 
        if ismatrix(I)
            I = cat(3,I,I,I);
        end
        
        % Resize the image as required for the CNN. 
        Iout = imresize(I, [227 227]);  
    end

%% 3 Crear división de train y test

[trainSamples, testSamples] = splitEachLabel(imds, 0.3, 'randomize');

%% 4 Aplicar CNN a las imagenes para obtener features

% Load pre-trained AlexNet
net = alexnet();
featureLayer = 'fc7';

% features = activations(net,X,layer,Name,Value)

trainFeatures = activations(net, trainSamples, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

%% 5 Entrenar SVM con trainFeatures

trainLabels = trainSamples.Labels;
classifier = fitcecoc(trainFeatures, trainLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%% 6 Test SVM classifier

% Extract test features using the CNN
testFeatures = activations(net, testSamples, featureLayer, 'MiniBatchSize',32);

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);

% Tabulate the results using a confusion matrix.
testLabels = testSamples.Labels;
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

% Compute accuracy
accuracy = sum(diag(confMat)) / sum(confMat(:))
end