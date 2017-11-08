% %Ex1
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

%Ex2
load spectra
whos NIR octane
X=NIR;
Y=octane;
[dummy,h] = sort(octane);
oldorder = get(gcf,'DefaultAxesColorOrder');
set(gcf,'DefaultAxesColorOrder',jet(60));
plot3(repmat(1:401,60,1)',repmat(octane(h),1,401)',NIR(h,:)');
set(gcf,'DefaultAxesColorOrder',oldorder);
% xlabel('Wavelength Index'); ylabel('Octane'); axis('tight');
% grid on

p = .7; 
% proportion of rows to select for training
N = size(X,1); % total number of rows
tf = false(N,1); % create logical index vector
tf(1:round(p*N))= true;
tf = tf(randperm(N)); % randomise order
Xtrain = X(tf,:);
Ytrain = Y(tf,:);
Xtest = X(~tf,:);
Ytest = Y(~tf,:);

%My code
betaMR=regress(Ytrain-mean(Ytrain),Xtrain); %MR coeficients betaMr*testX = y'

[a,b,c,d,betaPLS]=plsregress(Xtrain,Ytrain,2)  %PLS

[PCALoadings,PCAScores,PCAVar] = pca(Xtrain,'Economy',false);    %PCA
betaPCR = regress(Ytrain-mean(Ytrain), PCAScores(:,1:10))
betaPCR = PCALoadings(:,1:10)*betaPCR
betaPCR = [mean(Ytrain) - mean(Xtrain)*betaPCR; betaPCR]
% PLOT TRAIN DATARESULTS
figure
predMR = Xtrain*betaMR+mean(Ytrain);
predPLS = [ones(size(Xtrain,1),1) Xtrain]*betaPLS;
predPCR = [ones(size(Xtrain,1),1) Xtrain]*betaPCR;
plot(Ytrain,predMR,'bo',Ytrain,predPLS,'r^',Ytrain,predPCR,'go');
legend({'MR','PLS','PCR'})
xlabel('Original')
ylabel('Predicted')

