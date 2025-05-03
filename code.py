import numpy as np
import cv2
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = np.random.RandomState(random_state)
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _kmeans_plusplus(self, X):
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features), dtype=X.dtype)
        
        # First center
        center_id = self.random_state.choice(n_samples)
        centers[0] = X[center_id]
        
        # Initialize distances
        distances = np.full(n_samples, np.inf)
        for c in range(1, self.n_clusters):
            distances = np.minimum(distances, np.sum((X - centers[c-1])**2, axis=1))
            probs = distances / np.sum(distances)
            centers[c] = X[self.random_state.choice(n_samples, p=probs)]
        
        return centers

    def _lloyd_iter(self, X, centers):
        for i in range(self.max_iter):
            # Assign clusters
            distances = np.sum((X[:, None] - centers)**2, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Update centers
            new_centers = np.empty_like(centers)
            for j in range(self.n_clusters):
                cluster_points = X[labels == j]
                if cluster_points.shape[0] > 0:
                    new_centers[j] = cluster_points.mean(axis=0)
                else:
                    new_centers[j] = X[self.random_state.choice(X.shape[0])]
            
            # Check convergence
            shift = np.sum((new_centers - centers)**2)
            if shift <= self.tol:
                break
            centers = new_centers
        
        final_distances = np.sum((X[:, None] - centers)**2, axis=2)
        labels = np.argmin(final_distances, axis=1)
        inertia = np.sum(final_distances[np.arange(len(final_distances)), labels])
        
        return centers, labels, inertia, i + 1

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        best_inertia = np.inf
        
        for _ in range(self.n_init):
            if self.init == 'k-means++':
                centers = self._kmeans_plusplus(X)
            elif self.init == 'random':
                centers = X[self.random_state.choice(X.shape[0], self.n_clusters, False)]
            
            centers, labels, inertia, n_iter = self._lloyd_iter(X, centers)
            
            if inertia < best_inertia:
                self.cluster_centers_ = centers
                self.labels_ = labels
                self.inertia_ = inertia
                self.n_iter_ = n_iter
                best_inertia = inertia
        
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        distances = np.sum((X[:, None] - self.cluster_centers_)**2, axis=2)
        return np.argmin(distances, axis=1)

def process_image(image_path):
    # Load image and convert to LAB color space
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Extract and normalize the 'a' channel
    a_channel = lab[:, :, 1].astype(np.float64)
    a_channel = (a_channel - a_channel.mean()) / a_channel.std()
    
    # Reshape for clustering
    X = a_channel.reshape(-1, 1)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    labels = kmeans.predict(X).reshape(lab.shape[:2])
    
    # Create result image with black background
    result = np.zeros_like(image_rgb)
    
    # Find green cluster (lowest 'a' value in LAB space)
    green_cluster = np.argmax(kmeans.cluster_centers_)
    
    # Set detected numbers to bright green
    result[labels == green_cluster] = [0, 255, 0]  # RGB format
    
    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(image_rgb), plt.title('Original Image')
    plt.subplot(122), plt.imshow(result), plt.title('Numbers in Green')
    plt.show()

    # Print clustering details
    print("Cluster Centers:", kmeans.cluster_centers_.flatten())
    print("Cluster Sizes:", [np.sum(labels == i) for i in range(2)])
    print(f"Inertia: {kmeans.inertia_:.2f}")
    print(f"Iterations: {kmeans.n_iter_}")

# Run the processing
process_image('6.jpg')