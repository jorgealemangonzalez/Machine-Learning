function [ pattern ] = createTemplate( data , namePattern )
%CREATETEMPLATE Given the samples in the data matrix, create a template
%using the namePattern method. 
    switch namePattern
        case 'euclidean'
            %mean of the grayscale images
            pattern = squeeze(mean(data));
            
        case 'cityblock'
            pattern = squeeze(mean(data));
           
        case 'zVal'
            pattern.mean = squeeze(mean(data));
            pattern.std = squeeze(std(data));
            
        case 'K-NN'
            pattern = data;
            
    end
end

