import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def cluster(embedding, n_clusters, n_init=10, random_state=1, max_iter=200):
    scaled = StandardScaler().fit_transform(embedding)
    model = GaussianMixture(n_clusters, covariance_type="tied", n_init=n_init,
                            random_state=random_state, max_iter=max_iter)
    return model.fit_predict(scaled).astype(np.int32)


def evaluate(predicted, truth):
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    if predicted.shape != truth.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {truth.shape}")
    return {"ARI": adjusted_rand_score(truth, predicted) * 100,
            "NMI": normalized_mutual_info_score(truth, predicted) * 100}
