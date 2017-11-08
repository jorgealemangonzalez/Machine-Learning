%Useful functions:

%Ex1:
svmModel=fitcsvm(X,Y) %SVM
ldaModel=fitcdiscr(X,Y) %LDA

svmPrediction=predict(svmModel, Xtest)
ldaPrediction=predict(ldaModel, Xtest)

%ACCURACY

acc_svm=mean(strcmp(Ytest,svmPrediction))

%Ex2:

betaMR=regress(Y-mean(Y),X) %MR

%[...,betaPLS]=plsregress(X,Y,ndims)  %PLS

%PCR:
[PCALoadings,PCAScores,PCAVar] = pca(X,'Economy',false);
betaPCR = regress(y-mean(y), PCAScores(:,1:2));
betaPCR = PCALoadings(:,1:2)*betaPCR;
betaPCR = [mean(y) - mean(X)*betaPCR; betaPCR];


%Predict:
yPredMR= X*betaMR;
yPredPCR = [ones(n,1) X]*betaPCR;
yPredPLS=[ones(length_X,1) X]*betaPLS;