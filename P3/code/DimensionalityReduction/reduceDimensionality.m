function [ dataProjected, meanProjection, vectorsProjection ] = reduceDimensionality( data, drMethod, dimensions,labels  )
%REDUCEDIMENSIONALITY Reduce the dimensionality of the data contained in
%the matrix data using PCA or LDA.
    switch drMethod
        case 'PCA'
            [dataProjected mappingPCA]= compute_mapping(data,'PCA', dimensions);
            meanProjection = mappingPCA.mean;
            vectorsProjection = mappingPCA.M;
        case 'LDA'
            if(size(labels,1) < size(labels,2))
               labels = labels';
            end
            datadWithLabels = [labels data];
            [dataProjected mappingLDA]= compute_mapping(datadWithLabels,'LDA', dimensions);
            meanProjection = mappingLDA.mean;
            vectorsProjection = mappingLDA.M;
        case 'kernelPCAgaussian'
            [dataProjected mappingKernelPCA]= compute_mapping(data,'KernelPCA', dimensions, 'gauss');
%             mapping.X = X;
%             mapping.V = V;
%             mapping.invsqrtL = invsqrtL;
%             mapping.kernel = kernel;
%             mapping.param1 = param1;
%             mapping.param2 = param2;
        case 'kernelPCApolynomial'
            [dataProjected mappingKernelPCA]= compute_mapping(data,'KernelPCA', dimensions, 'poly');
    end

end

