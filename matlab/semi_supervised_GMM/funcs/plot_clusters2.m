function [] = plot_clusters2(mu, cov)
% Code written by Lawrence Bull
k = size(mu, 1);
% hold on
% plot the ellipses
theta = 0:0.1:2 * pi;
for i = 1:k
    [vec, val] = eig(cov(:,:,i));
    alph = atan(vec(2,1)/vec(1,1));
    
    for j = 2:2
        cx = j*val(1,1)^0.5*cos(theta);
        cy = j*val(2,2)^0.5*sin(theta);
        cr = [cos(alph), -sin(alph); sin(alph), cos(alph)]*[cx; cy];
        c = cr + mu(i,:)'; 
        plot(c(1,:), c(2,:), 'b--', 'LineWidth', 1)
    end
end
% plot the mean
% scatter(mu(:,1), mu(:,2), 80, 'b.', 'LineWidth', 2)
% hold off
end