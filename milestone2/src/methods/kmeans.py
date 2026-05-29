import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100, init="random"):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """

        ### WRITE YOUR CODE HERE
        self.K = K
        self.max_iters = max_iters
        self.init = init
        self.centers = None
        self.cluster_center_label = None
        self.final_inertia = None


    def init_centers(self, data):
        """
        Randomly pick K data points from the data as initial cluster centers.

        Arguments:
            data: array of shape (NxD) where N is the number of data points and D is the number of features (:=pixels).
            K: int, the number of clusters.
        Returns:
            centers: array of shape (KxD) of initial cluster centers
        """

        ### WRITE YOUR CODE HERE
        random_idx = np.random.choice(data.shape[0], self.K, replace=False)
        return data[random_idx]

    def init_centers_kmeanspp(self, data):
        """
        Initialize centers with KMeans++.

        Arguments:
            data: array of shape (N, D)
        Returns:
            centers: array of shape (K, D)
        """
        N, D = data.shape
        centers = np.zeros((self.K, D))

        first_idx = np.random.randint(N)
        centers[0] = data[first_idx]

        closest_sq_dist = np.sum((data - centers[0]) ** 2, axis=1)

        for k in range(1, self.K):
            total = np.sum(closest_sq_dist)
            if total <= 0 or not np.isfinite(total):
                next_idx = np.random.randint(N)
            else:
                probs = closest_sq_dist / total
                next_idx = np.random.choice(N, p=probs)

            centers[k] = data[next_idx]
            sq_dist_new_center = np.sum((data - centers[k]) ** 2, axis=1)
            closest_sq_dist = np.minimum(closest_sq_dist, sq_dist_new_center)

        return centers

    def compute_distance(self, data, centers):
        """
        Compute the euclidean distance between each datapoint and each center.

        Arguments:
            data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
            centers: array of shape (K, D), centers of the K clusters.
        Returns:
            distances: array of shape (N, K) with the distances between the N points and the K clusters.
        """

        ### WRITE YOUR CODE HERE
        N = data.shape[0]
        K = centers.shape[0]
        distances = np.zeros((N, K))
        for k in range(K):
            distances[:, k] = np.sqrt(np.sum((data - centers[k])**2, axis=1))
        return distances


    def find_closest_cluster(self, distances):
        """
        Assign datapoints to the closest clusters.

        Arguments:
            distances: array of shape (N, K), the distance of each data point to each cluster center.
        Returns:
            cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
        """

        ### WRITE YOUR CODE HERE
        return np.argmin(distances, axis=1)


    def compute_centers(self, data, cluster_assignments):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments:
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
            K: the number of clusters
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """

        ### WRITE YOUR CODE HERE
        centers = np.zeros((self.K, data.shape[1]))
        for k in range(self.K):
            centers[k] = np.mean(data[cluster_assignments == k], axis=0)
        return centers

    def compute_inertia(self, data, centers, assignments):
        """
        Compute the within-cluster sum of squares (inertia).

        Arguments:
            data: array of shape (N, D)
            centers: array of shape (K, D)
            assignments: array of shape (N,)
        Returns:
            inertia: float
        """
        diffs = data - centers[assignments]
        return np.sum(diffs ** 2)


    def k_means(self, data, max_iter=100):
        """
        Main K-Means algorithm that performs clustering of the data.

        Arguments:
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """

        ### WRITE YOUR CODE HERE
        if self.init == "random":
            centers = self.init_centers(data)
        elif self.init == "kmeans++":
            centers = self.init_centers_kmeanspp(data)
        else:
            raise ValueError(f"Unknown init mode: {self.init}. Use 'random' or 'kmeans++'.")
        for i in range(max_iter):
            old_centers = centers.copy()
            distances = self.compute_distance(data, centers)
            cluster_assignments = self.find_closest_cluster(distances)
            centers = self.compute_centers(data, cluster_assignments)
            if np.all(old_centers == centers):
                break
        distances = self.compute_distance(data, centers)
        cluster_assignments = self.find_closest_cluster(distances)
        return centers, cluster_assignments


    def assign_labels_to_centers(self, centers, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.

        Arguments:
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns:
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """

        ### WRITE YOUR CODE HERE
        K = centers.shape[0]
        cluster_center_label = np.zeros(K)
        for k in range(K):
            labels_in_k = true_labels[cluster_assignments == k]
            cluster_center_label[k] = np.argmax(np.bincount(labels_in_k.astype(int)))
        return cluster_center_label

    def predict_with_centers(self, data, centers, cluster_center_label):
        """
        Predict the label for data, given the cluster center and their labels.
        To do this, it first assign points in data to their closest cluster, then use the label
        of that cluster as prediction.

        Arguments:
            data: array of shape (N, D)
            centers: array of shape (K, D), cluster centers
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        Returns:
            new_labels: array of shape (N,), the labels assigned to each data point after clustering, via k-means.
        """

        ### WRITE YOUR CODE HERE
        distances = self.compute_distance(data, centers)
        cluster_assignments = self.find_closest_cluster(distances)
        return cluster_center_label[cluster_assignments]

    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ### WRITE YOUR CODE HERE
        centers, cluster_assignments = self.k_means(training_data, self.max_iters)
        cluster_center_label = self.assign_labels_to_centers(centers, cluster_assignments, training_labels)
        self.centers = centers
        self.cluster_center_label = cluster_center_label
        self.final_inertia = self.compute_inertia(training_data, centers, cluster_assignments)
        return self.predict_with_centers(training_data, centers, cluster_center_label)

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ### WRITE YOUR CODE HERE
        return self.predict_with_centers(test_data, self.centers, self.cluster_center_label)
