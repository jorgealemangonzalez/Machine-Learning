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
                [trainSamples, meanProjection, vectorsProjection] = reduceDimensionality( trainSamples, 'PCA', 530, labelsTrain);
                testSamples = testSamples * vectorsProjection;
            case 'minPCA'
                %In classify the number of samples of EACH GROUP must be
                %greater than the number of variables
                elements = zeros(7,1);
                for i = 1:size(labelsUsed(:))
                    elements(i) = sum(labelsTrain(:) == labelsUsed(i));
                end
                minElements = min(elements) - 1;
                
                [trainSamples, meanProjection, vectorsProjection] = reduceDimensionality( trainSamples, 'PCA', minElements, labelsTrain);
                testSamples = testSamples * vectorsProjection;
            case 'LDA'
                [dataProjected, meanProjection, vectorsProjectionPCA] = reduceDimensionality( trainSamples, 'PCA', 520, labelsTrain);
                [trainSamples, meanProjection, vectorsProjectionLDA] = reduceDimensionality( dataProjected, 'LDA', 6, labelsTrain);
                testSamples = testSamples * vectorsProjectionPCA * vectorsProjectionLDA;
            case 'kernelPCAgaussian'
                [trainSamples, meanProjection, vectorsProjection] = reduceDimensionality( trainSamples, 'kernelPCAgaussian', 530, labelsTrain);
                testSamples = testSamples * vectorsProjection;
            case 'kernelPCApolynomial'
                [trainSamples, ~, vectorsProjection] = reduceDimensionality( trainSamples, 'kernelPCApolynomial', 530, labelsTrain);
                testSamples = testSamples * vectorsProjection;
		%Check de compute_mapping function.

        end

        switch classificationMethod
            case 'example'
                % SAMPLE OF MATLAB's implementation of several classifiers
                knn = fitcknn(trainSamples, labelsTrain);
                estimatedLabels=knn.predict(testSamples);
               
            case 'Mahalanobis'
                estimatedLabels = classify(testSamples, trainSamples, labelsTrain, 'mahalanobis');
            case 'SVM'
                % TODO:
                % Train and classify with an implementation of SVM
                % HINT: check Matlab's svmtrain / svmclassify
                % REMEMBER: basic SVM is intended for binary classification. It MUST be extended to a 
                %  multiclass level, look for a strategy (data partition, iterative one-against-all, etc)
                nClasses = length(labelsUsed);
                %Buld as many models as classes 1vsAll
                for k=1:nClasses
                    cls = labelsUsed(k);
                    newLabelsTrain =(labelsTrain == cls);
                    SVMStructs(k) = svmtrain(trainSamples, newLabelsTrain);
                end
                
                %classify test cases
                for j=1:size(testSamples,1)
                    for k=1:nClasses
                        if(svmclassify(SVMStructs(k),testSamples(j,:))) 
                            break;
                        end
                    end
                    estimatedLabels(j) = labelsUsed(k);
                end
            case 'SVMgaussian'
                % TODO:
                % Train and classify with an implementation of SVM
                % HINT: check Matlab's svmtrain / svmclassify
                % REMEMBER: basic SVM is intended for binary classification. It MUST be extended to a 
                %  multiclass level, look for a strategy (data partition, iterative one-against-all, etc)
                nClasses = length(labelsUsed);
                %Buld as many models as classes 1vsAll
                for k=1:nClasses
                    cls = labelsUsed(k);
                    newLabelsTrain =(labelsTrain == cls);
                    SVMStructs(k) = svmtrain(trainSamples, newLabelsTrain, 'kernel_function', 'rbf');
                end
                
                %classify test cases
                for j=1:size(testSamples,1)
                    for k=1:nClasses
                        if(svmclassify(SVMStructs(k),testSamples(j,:))) 
                            break;
                        end
                    end
                    estimatedLabels(j) = labelsUsed(k);
                end    
            case 'SVMpolynomialKernel'
                % TODO:
                % Train and classify with an implementation of SVM
                % HINT: check Matlab's svmtrain / svmclassify
                % REMEMBER: basic SVM is intended for binary classification. It MUST be extended to a 
                %  multiclass level, look for a strategy (data partition, iterative one-against-all, etc)
                nClasses = length(labelsUsed);
                %Buld as many models as classes 1vsAll
                for k=1:nClasses
                    cls = labelsUsed(k);
                    newLabelsTrain =(labelsTrain == cls);
                    SVMStructs(k) = svmtrain(trainSamples, newLabelsTrain, 'kernel_function', 'polynomial');
                end
                
                %classify test cases
                for j=1:size(testSamples,1)
                    for k=1:nClasses
                        if(svmclassify(SVMStructs(k),testSamples(j,:))) 
                            break;
                        end
                    end
                    estimatedLabels(j) = labelsUsed(k);
                end
                
            case 'kernelSVMgaussian'
            case 'kernelSVMpolynomial'

        end

        %Create confusion matrix evaluating the templates with the test data
        confusionMatrix = confusionMatrix + confusionmat(estimatedLabels, labelsTest, 'ORDER', labelsUsed);
    end
    
    %get the total accuracy of the system
    accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));
end

