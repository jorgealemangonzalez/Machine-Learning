%a
p1 = [0,1];
p2 = [0,-1];
p3 = [1, 0];
p4 = [-1, 0];
[t1,r1] = cart2pol(p1(1),p1(2));
[t2,r2] = cart2pol(p2(1),p2(2));
[t3,r3] = cart2pol(p3(1),p3(2));
[t4,r4] = cart2pol(p4(1),p4(2));
observations = [t1,r1; t2,r2; t3,r3; t4,r4];
plot(observations(:,1), observations(:,2) ,'.','markers',50)
grid on
xlim([-2,4])
ylim([-2,4])
covariance = cov(observations)

%b
pca_coeff = pca(observations)
figure
plot(observations(:,1), observations(:,2) ,'.','markers',50)
grid on
xlim([-2,4])
ylim([-2,4])
hold on
plot([0,pca_coeff(1,1)],[0,pca_coeff(1,2)],'->')
plot([0,0],pca_coeff(2,:),'->')

%c
observations_pca = observations * pca_coeff
% coeff => identity, no changes on the values
figure
plot(observations_pca(:,1), observations_pca(:,2) ,'.','markers',50)
grid on
xlim([-2,4])
ylim([-2,4])

%d
projected_observations_pca = observations * [pca_coeff(1,:);[0,0]]
figure
plot(observations(:,1), observations(:,2) ,'.','markers',50)
hold on
plot(projected_observations_pca(:,1), projected_observations_pca(:,2) ,'.','markers',50)
grid on
xlim([-2,4])
ylim([-2,4])
