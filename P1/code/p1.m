
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% TEST DIFFERENT TEMPLATES METHODS %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%[accuracy confMatrix] = testMethod(imagesData , labels, emotionsUsed ,  'grayscaleMean', 'euclidean', indexesCrossVal)
%[accuracy confMatrix] = testMethod(imagesData , labels, emotionsUsed ,  'chamferDistance', 'euclidean', indexesCrossVal)
[accuracy confMatrix] = testMethod(imagesData , labels, emotionsUsed ,  'zVal', 'euclidean', indexesCrossVal)

%data1 = imagesData(1,:,:)