function [ estimatedLabels ] = classifyWithTemplateMatching( templates , testData , method, errorMeasure,emotions)
%CLASSIFYWITHTEMPLATEMATCHING Given a set of templates and a test dataset,
%this function estimates the labels of each sample in the test dataset
%comparing it with each of the templates.
    
    %Convert all the images in the testData into a chamfer distance images
    if(strcmp(method,'chamferMean')==1)
        for i = 1:size(testData,1)
            image = squeeze(testData(i,:,:));
            testData(i,:,:) = bwdist(edge(image,'canny',0.4));
        end
    end

    %init the variable where the estimated labels will be stored
    estimatedLabels = zeros(1,size(testData,1));
    %get the number of templates we are going to evaluate
    numTemplates = size(templates,1);
    
    if(strcmp(method,'hist')==1)
        totalSamples = 0;
        for i = 1:numTemplates
            totalSamples = totalSamples + templates(i).samples;
        end
    end
    
    %Iterate over all the test data
    for i = 1:size(testData,1)
        %get the current sample we want to evaluate
        currentSample = squeeze(testData(i,:,:));
        %init the similarity score for each template with the current
        %sample
        templateScore = zeros(1,numTemplates);
        if strcmp(errorMeasure, 'K-NN')
            dx = 1;% Distance index K-NN
            K = 10;
        end
        for e = 1:numTemplates
            %get the current template
            currentTemplate = squeeze(templates(e,:,:));
            if strcmp(errorMeasure, 'K-NN')
               for r = 1:size(currentTemplate.raw,1)
                   temp = squeeze(currentTemplate.raw(r,:,:));
                   distances(dx,:,:) = [pdist2(currentSample(:)', temp(:)', 'euclidean') e]; % #samples x 1 x 2  -> 1x2 = [distance class]
                   dx = dx +1;
               end
            end
               
            %get the similarity score of the pattern with the given sample
            %and store into templateScore variable
            switch errorMeasure
                case 'euclidean'
                    templateScore(e) = pdist2(currentSample(:)', currentTemplate(:)','euclidean');
                case 'cityblock'
                    templateScore(e) = pdist2(currentSample(:)', currentTemplate(:)','cityblock');
                case 'zVal'
                    z = ( currentSample(:) - currentTemplate.mean(:) ) ./ currentTemplate.std(:);
                    z2 = z.^4;      %More importance to higher errors
                    templateScore(e) = sum(z2);   % <-- We want to minimize this value
                    %Z values might be worst than euclidean distance to
                    %mean because the distribution is not GAUSIAN
                    
                case 'bayesian'
                    cs = currentSample(:);
                    prior = currentTemplate.samples / totalSamples;
                    
                    for csi = 1:size(cs)
                        templateScore(e) = templateScore(e) + (currentTemplate.hist( ceil(( (cs(csi)+1)/256)*15) , csi) * prior);
                    end
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

