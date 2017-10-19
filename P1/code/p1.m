
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% P1 - RECONEIXEMENT DE PATRONS %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%    TEMPLATE MATCHING          %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% EXTRACT DATA %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[imagesData shapeData labels] = extractData('../CKDB', emotionsUsed);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% DIVIDE DATA (TRAIN/TEST) WITH CROSS VALIDATION  %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = 3;
indexesCrossVal = crossvalind('Kfold',size(imagesData,1),K);
%load indexesCrossVal.mat
%load indexesCrossVal10.mat
%indexesCrossVal = indexesCrossVal10;
lastIndexes = indexesCrossVal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% TEST DIFFERENT TEMPLATES METHODS %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tiempo_inicio = cputime;
% [accuracy confMatrix]=testMethod(imagesData , labels, emotionsUsed ,  'grayscaleMean', 'euclidean', indexesCrossVal)
% ALLRESULTS('CKDBVeryHard_imagesData_grayscaleMean_euclidean') = {accuracy,confMatrix,cputime - tiempo_inicio};
% tiempo_inicio = cputime;
% [accuracy confMatrix] = testMethod(imagesData , labels, emotionsUsed ,  'grayscaleMean', 'cityblock', indexesCrossVal)
% ALLRESULTS('CKDBVeryHard_imagesData_grayscaleMean_cityblock') = {accuracy,confMatrix,cputime - tiempo_inicio};
% tiempo_inicio = cputime;
% [accuracy confMatrix] = testMethod(imagesData , labels, emotionsUsed ,  'chamferMean', 'euclidean', indexesCrossVal)
% ALLRESULTS('CKDBVeryHard_imagesData_chamferMean_euclidean') = {accuracy,confMatrix,cputime - tiempo_inicio};
% tiempo_inicio = cputime;
% [accuracy confMatrix] = testMethod(imagesData , labels, emotionsUsed ,  'grayscaleMeanDeviation', 'zVal', indexesCrossVal)
% ALLRESULTS('CKDBVeryHard_imagesData_grayscaleMeanDeviation_zVal') = {accuracy,confMatrix,cputime - tiempo_inicio};
tiempo_inicio = cputime;
[accuracy confMatrix] = testMethod(imagesData , labels, emotionsUsed ,  'hist', 'bayesian', indexesCrossVal)
ALLRESULTS('CKDB_imagesData_hist_bayesian') = {accuracy,confMatrix,cputime - tiempo_inicio};
% tiempo_inicio = cputime;
% [accuracy confMatrix] = testMethod(imagesData , labels, emotionsUsed ,  'raw', 'K-NN', indexesCrossVal)
% ALLRESULTS('CKDBVeryHard_imagesData_raw_K-NN') = {accuracy,confMatrix,cputime - tiempo_inicio};
% tiempo_inicio = cputime;
% [accuracy confMatrix] = testMethod(shapeData , labels, emotionsUsed ,  'grayscaleMean', 'euclidean', indexesCrossVal)
% ALLRESULTS('CKDBVeryHard_shapeData_grayscaleMean_euclidean') = {accuracy,confMatrix,cputime - tiempo_inicio};
% tiempo_inicio = cputime;
% [accuracy confMatrix] = testMethod(shapeData , labels, emotionsUsed ,  'grayscaleMean', 'cityblock', indexesCrossVal)
% ALLRESULTS('CKDBVeryHard_shapeData_grayscaleMean_cityblock') = {accuracy,confMatrix,cputime - tiempo_inicio};
% tiempo_inicio = cputime;
% [accuracy confMatrix] = testMethod(shapeData , labels, emotionsUsed ,  'grayscaleMeanDeviation', 'zVal', indexesCrossVal)
% ALLRESULTS('CKDBVeryHard_shapeData_grayscaleMeanDeviation_zVal') = {accuracy,confMatrix,cputime - tiempo_inicio};
tiempo_inicio = cputime;
[accuracy confMatrix] = testMethod(shapeData , labels, emotionsUsed ,  'hist', 'bayesian', indexesCrossVal)
ALLRESULTS('CKDB_shapeData_hist_bayesian') = {accuracy,confMatrix,cputime - tiempo_inicio};
% tiempo_inicio = cputime;
% [accuracy confMatrix] = testMethod(shapeData , labels, emotionsUsed ,  'raw', 'K-NN', indexesCrossVal)
%    ALLRESULTS('CKDBVeryHard_shapeData_raw_K-NN') = {accuracy,confMatrix,cputime - tiempo_inicio};