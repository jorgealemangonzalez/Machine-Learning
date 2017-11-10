%% Ex1
% load ionosphere
% p = .7; % proportion of rows to select for training
% N = size(X,1); % total number of rows
% tf = false(N,1); % create logical index vector
% tf(1:round(p*N))= true;
% tf = tf(randperm(N)); % randomise order
% X = X(:,[1 3:end]);
% Xtrain = X(tf,:);
% Ytrain = Y(tf,:);
% Xtest = X(~tf,:);
% Ytest = Y(~tf,:);
% 
% svmModel=fitcsvm(Xtrain,Ytrain); %SVM
% ldaModel=fitcdiscr(Xtrain,Ytrain); %LDA
% 
% svmPrediction=predict(svmModel, Xtest);
% ldaPrediction=predict(ldaModel, Xtest);
% 
% acc_svm=mean(strcmp(Ytest,svmPrediction))
% acc_lda=mean(strcmp(Ytest,ldaPrediction))

%% Ex2
load spectra
whos NIR octane
X=NIR;
Y=octane;
N = size(X,1); % total number of rows
% [dummy,h] = sort(octane);
% oldorder = get(gcf,'DefaultAxesColorOrder');
% set(gcf,'DefaultAxesColorOrder',jet(60));
% plot3(repmat(1:401,60,1)',repmat(octane(h),1,401)',NIR(h,:)');
% set(gcf,'DefaultAxesColorOrder',oldorder);
% xlabel('Wavelength Index'); ylabel('% Octane'); axis('tight');
% grid on

% p = .7; 
% % proportion of rows to select for training
% tf = false(N,1); % create logical index vector
% tf(1:round(p*N))= true;
% tf = tf(randperm(N)); % randomise order
% Xtrain = X(tf,:);
% Ytrain = Y(tf,:);
% Xtest = X(~tf,:);
% Ytest = Y(~tf,:);
% 
% % My code
% Ymean = mean(Ytrain);
% betaMLR=regress(Ytrain-Ymean,Xtrain); %MLR coeficients betaMr*testX = y'
% 
% [a,b,c,d,betaPLS,PCTVar]=plsregress(Xtrain,Ytrain,10);  %PLS
% 
% [PCALoadings,PCAScores,PCAVar] = pca(Xtrain,'Economy',false);    %PCA
% betaPCR = regress(Ytrain-Ymean, PCAScores(:,1:10));
% betaPCR = PCALoadings(:,1:10)*betaPCR;
% betaPCR = [Ymean - mean(Xtrain)*betaPCR; betaPCR];
% 
% % Plot train and test results
% figure
% predMLR_train = Xtrain*betaMLR+Ymean;
% predMLR_test = Xtest*betaMLR+Ymean;
% predPLS_train = [ones(size(Xtrain,1),1) Xtrain]*betaPLS;
% predPLS_test = [ones(size(Xtest,1),1) Xtest]*betaPLS;
% predPCR_train = [ones(size(Xtrain,1),1) Xtrain]*betaPCR;
% predPCR_test = [ones(size(Xtest,1),1) Xtest]*betaPCR;
% plot(Ytest,predMLR_test,'b^',Ytest,predPLS_test,'r^',Ytest,predPCR_test,'g^');
% hold on
% plot(Ytrain,predMLR_train,'b.',Ytrain,predPLS_train,'r.',Ytrain,predPCR_train,'g.');
% plot([Ytrain' Ytest']',[Ytrain' Ytest']','k-')
% legend({'MLR test','PLS test','PCR test', 'MLR train', 'PLS train', 'PCR train'})
% xlabel('Original')
% ylabel('Predicted')
% grid on
% 
% %Plot of variance by components
% figure
% plot(1:10,cumsum(100*PCTVar(2,:)),'-bo');   %PLS
% xlabel('Number of PLS components');
% ylabel('Percent Variance');

% Cross validation of models
K = 3;
indices = crossvalind('Kfold', N, K);

errorMLR = zeros(10,1);
errorPLS = zeros(10,1);
errorPCR = zeros(10,1);

for C=1:10 %components

for i=1:K
    test = (indices == i);
    train = ~test;
    
    Xtest = X(test,:);
    Ytest = Y(test,:);
    Xtrain = X(train,:);
    Ytrain = Y(train,:);
    
    n = size(Xtest, 1);
    
    %Create models
    Ymean = mean(Ytrain);
    
    betaMLR=regress(Ytrain-Ymean,Xtrain);   %MLR
    [a,b,c,d,betaPLS,PCTVar]=plsregress(Xtrain,Ytrain,C);  %PLS
    [PCALoadings,PCAScores,PCAVar] = pca(Xtrain,'Economy',false);    %PCA
    betaPCR = regress(Ytrain-Ymean, PCAScores(:,1:C));
    betaPCR = PCALoadings(:,1:C)*betaPCR;
    betaPCR = [Ymean - mean(Xtrain)*betaPCR; betaPCR];
    
    %Test models
    testMLR = Xtest * betaMLR + Ymean;
    testPLS = [ones(size(Xtest,1),1) Xtest]*betaPLS;
    testPCR = [ones(size(Xtest,1),1) Xtest]*betaPCR;
    
%     figure
%     plot(Ytest,testMLR,'b^',Ytest,testPLS,'r^',Ytest,testPCR,'g^');
%     hold on
%     plot([Ytrain' Ytest']',[Ytrain' Ytest']','k-')
%     legend({'MLR test','PLS test','PCR test'})
%     xlabel('Original')
%     ylabel('Predicted')
%     grid on
    
    errorMLR(C) = errorMLR(C) + (sum(abs(testMLR - Ytest) ./ Ytest)/n);
    errorPLS(C) = errorPLS(C) + (sum(abs(testPLS - Ytest) ./ Ytest)/n);
    errorPCR(C) = errorPCR(C) + (sum(abs(testPCR - Ytest) ./ Ytest)/n);
end

errorMLR(C) = errorMLR(C) / K;
errorPLS(C) = errorPLS(C) / K;
errorPCR(C) = errorPCR(C) / K;
end

errorMLR = sum(errorMLR)/C  %It doesn't depend on components, so we get 1 unic error with the mean
errorPLS
errorPCR

figure
plot(1:10, errorPLS, '-ro', 1:10, errorPCR, '-go')
legend({'PLS Error', 'PCR Error'})