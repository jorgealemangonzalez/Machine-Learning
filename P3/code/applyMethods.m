function [ accuracy confusionMatrix ] = applyMethods(data, labels, labelsUsed, indexesCrossVal, classificationMethod,dimensionalityReductionMethod)
    
    % ApplyMethods
    
    % This function estimates the accuracy and confusion over the dataset samples in data by using Cross Validation 
    % the input template matching method and the error measure method.
    
    % INPUTS:
    % data: NxDxD , matrix with the images or shape data of the dataset
    % labels: 1xN vector with the emotion labels for each sample in the matrix data.
    % labelsUsed: labels used to train and classify.
    % templateMethod: string with the method used to generate the template.
    % errorMeasuse: string with the error measure method used to compare the template with the samples.
    % indexesCrossVal: indexes used to perform the performance evaluation with cross validation

    indexes = indexesCrossVal;
    K = max(indexes);

    confusionMatrix = zeros(numel(labelsUsed));

    for k = 1:K
        display(['Testing data subset: ' num2str(k) '/' num2str(K)]);
        %get train and test dataset with the indexes obtained with the KFold
        %cross validation
        trainSamples = data(indexes~=k,:,:);
        labelsTrain  = labels(indexes~=k);
        
        testSamples  = data(indexes==k,:,:);
        labelsTest   = labels(indexes==k);

        switch dimensionalityReductionMethod
            case 'PCA'
                %CODE HERE
                
            case 'LDA'
                %CODE HERE

            case 'kernelPCA'    
                %CODE HERE
		%Check de compute_mapping function.

        end



        switch classificationMethod
            case 'example'
                % SAMPLE OF MATLAB's implementation of several classifiers
                knn = fitcknn(trainSamples, labelsTrain);
                estimatedLabels=knn.predict(testSamples);
                
                
            case 'SVM'
                % TODO:
                % Train and classify with an implementation of SVM
                % HINT: check Matlab's svmtrain / svmclassify
                % REMEMBER: basic SVM is intended for binary classification. It MUST be extended to a 
                %  multiclass level, look for a strategy (data partition, iterative one-against-all, etc)
                
            case 'Mahalanobis'

            case 'kernelSVM'     

        end

        %Create confusion matrix evaluating the templates with the test data
        confusionMatrix = confusionMatrix + confusionmat(estimatedLabels, labelsTest, 'ORDER', labelsUsed);
    end
    
    %get the total accuracy of the system
    accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));
end
