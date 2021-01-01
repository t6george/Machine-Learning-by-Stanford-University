function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

[m n] = size(X);

centroids = zeros(K, n);


freq = zeros(K,1);

for i = 1:m
    freq(idx(i)) = freq(idx(i)) + 1;
    centroids(idx(i),:) = centroids(idx(i),:) + X(i,:);
end

centroids = centroids ./ reshape(repmat(freq, n, 1), size(centroids));

end


