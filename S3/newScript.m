K = [2 4 5 10 20];
for k=K
    figure();
    [idx,C] = kmeans(X,k);
    gscatter(X(:,1),X(:,2),idx)
    hold on
    plot(C(:,1), C(:,2), 'kx', 'MarkerSize',15,'LineWidth',3);
end