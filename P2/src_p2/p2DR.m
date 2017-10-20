
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% P2 - RECONEIXEMENT DE PATRONS  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%    REDUCCIï¿½ DE DIMENSIONALITAT %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


%%GSCATTER 3 example. Visualize the first three coordiantes of the data.
%%You can remove this after understanding it! :)
% Basic data plot
% figure
% gscatter3(grayscaleFeatures(:,1),grayscaleFeatures(:,2),grayscaleFeatures(:,3),stringLabels,7)
% [dataProjected, meanProjection, vectorsProjection] = reduceDimensionality( grayscaleFeatures, 'PCA', 3, stringLabels);

% Exercice 1
% figure
% gscatter3(dataProjected(:,1),dataProjected(:,2),dataProjected(:,3),stringLabels,7)

% Exercice 3 
% DIMMENSIONS = [2 5 10 50 100 300 500];
% image1 = grayscaleFeatures(1,:);
% imagesc(reshape(image1,128,128));
% for i = 1:size(DIMMENSIONS,2)
%     dims = DIMMENSIONS(i);
%     [dataProjected, meanProjection, vectorsProjection] = reduceDimensionality( grayscaleFeatures, 'PCA', dims, stringLabels);
%     
%     projImg = dataProjected(1,:);
%     repImg = reprojectData(projImg, meanProjection, vectorsProjection);
%     figure;
%     imagesc(reshape( repImg, 128, 128))
% end
% [dataProjected, meanProjection, vectorsProjection] = reduceDimensionality( grayscaleFeatures, 'PCA', 3, stringLabels);

% Exercice 4 -> Solución: 75 dimensiones
% DIMMENSIONS = [2 5 10 50 100 300 500];
% errors = zeros(size(DIMMENSIONS,2));
% for i = 1:size(DIMMENSIONS,2)
%     dims = DIMMENSIONS(i);
%     [dataProjected, meanProjection, vectorsProjection] = reduceDimensionality( grayscaleFeatures, 'PCA', dims, stringLabels);
%     
%     totalErr = 0;
%     for j = 1:size(dataProjected,1)
%         projImg = dataProjected(j,:);
%         repImg = reprojectData(projImg, meanProjection, vectorsProjection);
%         err = immse(repImg, grayscaleFeatures(j,:));
%         totalErr = totalErr + err;
%     end
%     errors(i) = totalErr;
% end
% plot(DIMMENSIONS,errors);
% xlabel('Dimension reduction')
% ylabel('Square error sum')

% Exercice 5
% [dataProjected, meanProjection, vectorsProjection] = reduceDimensionality( grayscaleFeatures, 'PCA', 75, labels);
% [dataProjected, meanProjection, vectorsProjection] = reduceDimensionality( dataProjected, 'LDA', 3, labels);
% gscatter3(dataProjected(:,1),dataProjected(:,2),dataProjected(:,3),stringLabels,7)



%%%%%%%%%%%%%%% DIVIDE DATA (TRAIN/TEST) WITH CROSS VALIDATION  %%%%%%%%%
K = 6;
indexesCrossVal = crossvalind('Kfold',size(imagesData,1),K);

%%%%%%%  EXAMPLE OF CLASSIFYING THE EXPRESSION USING TEMPLATE  MATCHING %%%%
[ACC CONF] = testTemplateMatchingWithDR( grayscaleFeatures , labels, emotionsUsed , 'euclidean', indexesCrossVal )
% [ACC CONF] = testTemplateMatchingWithDR( grayscaleFeatures , labels, ...
% emotionsUsed , 'K-NN', indexesCrossVal ) %-> 0.8695