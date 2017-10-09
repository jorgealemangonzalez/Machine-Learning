function [ pattern ] = createTemplate( data , namePattern )
%CREATETEMPLATE Given the samples in the data matrix, create a template
%using the namePattern method. 
    switch namePattern
        case 'grayscaleMean'
            %mean of the grayscale images
            pattern = squeeze(mean(data));
            
        case 'chamferDistance'
            %binary contourn of images
            
            
            data2 = zeros(size(data));
            for i = 1:size(data,1)
                size(squeeze(data(i,:,:)))
                data2(i) = conv2([-1,-1,-1;-1,8,-1;-1,-1,-1], squeeze(data(i,:,:)));
            end
            
            pattern = squeeze(mean(data2));
            
        case 'zVal'
            meanData = squeeze(mean(data));
            varianceData = squeeze(std(data));
            
            pattern = cell(size(meanData));
            for i = 1:size(meanData,1)
                for j = 1:size(meanData,2)
                    pattern{i,j} = @(x) abs(x - meanData(i,j) / sqrt(varianceData(i,j)));     %   <-- anonymous function
                end
            end
            
        case 'gaussian'
            
                
    end
end

