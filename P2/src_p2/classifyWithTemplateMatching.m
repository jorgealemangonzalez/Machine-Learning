function [ estimatedLabels ] = classifyWithTemplateMatching( templates , testData , errorMeasure,emotions)
%CLASSIFYWITHTEMPLATEMATCHING Given a set of templates and a test dataset,
%this function estimates the labels of each sample in the test dataset
%comparing it with each of the templates.
   

    %init the variable where the estimated labels will be stored
    estimatedLabels = zeros(1,size(testData,1));
    %get the number of templates we are going to evaluate
    numTemplates = size(templates,2);
    
    %Iterate over all the test data
    for i = 1:size(testData,1)
        if strcmp(errorMeasure, 'K-NN')
            dx = 1;% Distance index K-NN
            K = 10;
        end
        %get the current sample we want to evaluate
        currentSample = squeeze(testData(i,:,:));
        %init the similarity score for each template with the current
        %sample
        templateScore = zeros(1,numTemplates);
        for e = 1:numTemplates
            
            if strcmp(errorMeasure, 'K-NN')
                %get the current template
                currentTemplate = templates{e};
               for r = 1:size(currentTemplate,1)
                   temp = squeeze(currentTemplate(r,:));
                   distances(dx,:,:) = [pdist2(currentSample(:)', temp(:)', 'euclidean') e]; % #samples x 1 x 2  -> 1x2 = [distance class]
                   dx = dx +1;
               end
            else
                %get the current template
                currentTemplate = templates{e};
            end
            
            %get the similarity score of the pattern with the given sample
            %and store into templateScore variable
            switch errorMeasure
                case 'euclidean'
                    templateScore(e) = pdist2(currentSample(:)', currentTemplate(:)','euclidean');
            end
        end      
        if strcmp(errorMeasure, 'K-NN')
            [D I] = sort(distances(:,:,1));    
            distances(:,:,1) = D;
            distances(:,:,2) = distances(I,:,2);
            results = distances(1:10,1,2);   % recover classes
            selectedClass = mode(results);
            templateScore = zeros(1,numTemplates);
            templateScore(selectedClass) = -1; % Min score to selected class
        end
        %get the label with the minimum similarity score and assign it to
        %the current sample
        estimatedLabels(i) = emotions(find(templateScore==min(templateScore),1));
        
    end
end

