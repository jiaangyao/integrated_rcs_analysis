from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
# from sklearn_extra.cluster import KMedoids
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.distributions import LogNormal, Normal
from sklearn.linear_model import LogisticRegression

import numpy as np
import torch

from model.base import BaseModel
from scipy.stats import entropy

class CustomClusteringClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, clustering_algorithm='kmeans', classifier='lda', n_clusters=2, random_state=42, eps=0.5, max_iter=100):
        self.clustering_algorithm = clustering_algorithm.lower()
        self.classifier = classifier.lower()
        self.n_clusters = n_clusters
        self.random_state = random_state

        if self.clustering_algorithm == 'kmeans':
            self.cluster_model = KMeans(n_clusters=n_clusters, n_init='auto', random_state=random_state)
        elif self.clustering_algorithm == 'agglomerative':
            self.cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
        elif self.clustering_algorithm == 'gmm' or self.clustering_algorithm == 'loggmm':
            self.scaler = StandardScaler()
            self.cluster_model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        elif self.clustering_algorithm == 'dbscan':
            self.cluster_model = DBSCAN(eps=eps)
        elif self.clustering_algorithm == 'loggmm_pomegranate':
            # Defer initialization until fit is called
            self.cluster_model = None
        # elif self.clustering_algorithm == 'kmedoids':
        #     self.cluster_model = KMedoids(n_clusters=n_clusters, random_state=random_state)
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.clustering_algorithm}")

        if self.classifier == 'lda' or self.classifier == 'lineardiscriminantanalysis':
            self.classifier_model = LinearDiscriminantAnalysis()
        elif self.classifier == 'logistic_regression' or self.classifier == 'logisticregression' or self.classifier == 'lr' or self.classifier == 'logistic':
            self.classifier_model = LogisticRegression(random_state=random_state, max_iter=max_iter)
        else:
            raise ValueError(f"Unknown classifier: {self.classifier}")

        self.cluster_labels = None
        self.classes_ = None
        self.class_labels = None
        self.cluster_to_label = None
        
    
    def _ensure_unique_cluster_to_label(self, cluster_centers, label_means, distances):
        # Initialize a list to store the final assignments
        cluster_to_label = [-1] * len(cluster_centers)

        # Keep track of assigned labels
        assigned_labels = set()

        # Iterate to assign each cluster the closest label
        for _ in range(len(cluster_centers)):
            min_distance = float('inf')
            min_cluster_index = -1
            min_label_index = -1
            
            # Find the closest unassigned label for any cluster
            for cluster_index in range(len(cluster_centers)):
                if cluster_to_label[cluster_index] == -1:  # If cluster is not yet assigned a label
                    for label_index in range(len(label_means)):
                        if label_index not in assigned_labels:
                            if distances[cluster_index][label_index] < min_distance:
                                min_distance = distances[cluster_index][label_index]
                                min_cluster_index = cluster_index
                                min_label_index = label_index
            
            # Assign the closest label to the cluster
            cluster_to_label[min_cluster_index] = min_label_index
            assigned_labels.add(min_label_index)

        # Convert cluster_to_label to a numpy array if needed
        return np.array(cluster_to_label)
        
        
    def fit(self, X, y):
        # Perform clustering
        self.class_labels = y
        
        if self.clustering_algorithm == 'loggmm_pomegranate':
            # Convert to torch tensor if not already
            # TODO: This throws often does not find multiple distributions...
            # TODO: Therefore, does not work very well...
            if not isinstance(X, torch.Tensor):
                X_torch = torch.from_numpy(X).float()
            else:
                X_torch = X.float()
                
            # Initialize the model now that we know the dimensions
            if self.cluster_model is None:
                d = X.shape[1]  # Get number of features
                np.random.seed(self.random_state)
                dists = [LogNormal(
                    means=torch.ones(d),  
                    covs=torch.eye(d),  # Use identity matrix for covariance: (d, d)
                    covariance_type='full'  # Specify full covariance type
                ) for _ in range(self.n_clusters)]
                self.cluster_model = GeneralMixtureModel(
                    distributions=dists,
                    priors=torch.ones(self.n_clusters)/self.n_clusters
                )
                
            self.cluster_model.fit(X_torch)
            self.cluster_labels = self.cluster_model.predict(X_torch)
        else:
            self.cluster_model.fit(X)
            if self.clustering_algorithm == 'gmm':
                self.cluster_labels = self.cluster_model.predict(X)
            elif self.clustering_algorithm == 'loggmm':
                X_scaled = self.scaler.fit_transform(X)
                # Shift to ensure positivity
                min_vals = np.min(X_scaled, axis=0, keepdims=True)
                X_shifted = X_scaled - min_vals + 1.0
                self.cluster_labels = self.cluster_model.predict(np.log(X_shifted))
            else:
                self.cluster_labels = self.cluster_model.labels_

        # Determine the cluster means
        cluster_centers = np.array([X[self.cluster_labels == i].mean(axis=0) for i in range(self.n_clusters)])
        
        # Determine the label means
        unique_labels = np.unique(y)
        self.classes_ = unique_labels
        label_means = [np.mean(X[y == label], axis=0) for label in unique_labels]

        # Reassign the cluster labels to the appropriate label number
        distances = np.array([[np.linalg.norm(cluster_center - label_mean) for label_mean in label_means] for cluster_center in cluster_centers])
        cluster_to_label = np.argmin(distances, axis=1)
        
        # Sometimes, multiple clusters are closest to a single label. In this case, we need to ensure that each label is assigned to a unique cluster.
        if len(np.unique(cluster_to_label)) < len(self.classes_):
            cluster_to_label = self._ensure_unique_cluster_to_label(cluster_centers, label_means, distances)
            
        self.cluster_to_label = cluster_to_label

        new_y = np.array([cluster_to_label[label] for label in self.cluster_labels])
        
        # Update cluster labels to reflect the new label assignments, i.e. get cluster labels consistent with original label numbers and corresponding distributions
        self.cluster_labels = new_y
        
        # Train the classifier on the new labels
        self.classifier_model.fit(X, new_y)

        print("break")

    def predict(self, X):
        return self.classifier_model.predict(X)

    def predict_proba(self, X):
        return self.classifier_model.predict_proba(X)

    def score(self, X, y):
        return self.classifier_model.score(X, y)
    
    # Get the distribution of labels per cluster. 
    # I.e. for each cluster (corresponding to rows in the return matrix), the number of samples in each label class. Returns a (num_clusters, num_classes) array.
    def label_dist_per_cluster(self):
        cluster_counts = np.zeros((self.n_clusters, len(self.classes_)))
        for cluster in range(self.n_clusters):
            cluster_indices = np.where(self.cluster_labels == cluster)
            cluster_labels = self.class_labels[cluster_indices]
            label_counts = np.array([np.sum(cluster_labels == i) for i in self.classes_]) # np.bincount(cluster_labels)
            cluster_counts[cluster, :] = label_counts
        return cluster_counts
    
    # Get the entropy of the label distribution per cluster. Returns a (num_clusters,) array, each element indicating the entropy of the label distribution in a cluster.
    # E.g. [(entroy(cluster_0), entropy(cluster_1), ...)]
    def cluster_entropy(self):
        try:
            cluster_counts = self.label_dist_per_cluster()
            cluster_counts = cluster_counts / cluster_counts.sum(axis=1, keepdims=True)
            cluster_entropy = entropy(cluster_counts, axis=1, base=2)
            return cluster_entropy
        except:
            print("Error: Cluster entropy calculation failed. Could be due to empty clusters, which may occur if number of clusters is greater than class labels.")
            print(f"Cluster counts: {cluster_counts}")
            return np.nan
        

class ClusterClassificationModel(BaseModel):
    def __init__(self, **model_kwargs):
        self.model_kwargs = model_kwargs
        super().__init__(CustomClusteringClassifier(**model_kwargs))

    def override_model(self, model_kwargs):
        self.model_kwargs = model_kwargs
        self.model = CustomClusteringClassifier(**model_kwargs)

    def reset_model(self):
        self.model = CustomClusteringClassifier(**self.model_kwargs)
