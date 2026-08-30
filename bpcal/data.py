import numpy as np
from sklearn.decomposition import PCA


def normalize(counts, target_sum=1e4):
    counts = np.asarray(counts, dtype=np.float64)
    lib = counts.sum(1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(counts / lib * target_sum)


def preprocess(counts, n_pcs=30, random_state=0):
    x = normalize(counts)
    return PCA(min(n_pcs, *x.shape), random_state=random_state).fit_transform(x)


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    labels = d["labels"] if "labels" in d.files else None
    return d["counts"], d["coords"], labels


def load_loom(path, region_key="Region", valid_key="Valid", exclude="Excluded"):
    import h5py

    with h5py.File(path, "r") as f:
        counts = f["matrix"][:].T.astype(np.float32)
        coords = np.column_stack([f["col_attrs"]["X"][:],
                                  f["col_attrs"]["Y"][:]]).astype(np.float64)
        region = np.array([r.decode() if isinstance(r, bytes) else r
                           for r in f["col_attrs"][region_key][:]])
        valid = f["col_attrs"][valid_key][:]

    keep = (valid == 1) & (region != exclude)
    domains = np.unique(region[keep])
    return counts[keep], coords[keep], np.searchsorted(domains, region[keep])


def load_h5ad(path, coord_key="spatial", label_key=None):
    import anndata

    adata = anndata.read_h5ad(path)
    counts = adata.layers.get("counts", adata.X)
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    coords = np.asarray(adata.obsm[coord_key], dtype=np.float64)

    if label_key is None:
        return counts, coords, None

    import pandas as pd

    keep = adata.obs[label_key].notna().to_numpy()
    labels = pd.Categorical(adata.obs[label_key][keep].astype(str)).codes
    return counts[keep], coords[keep], labels.astype(np.int64)


def load_visium(directory, label_file=None):
    import scanpy as sc

    adata = sc.read_visium(directory)
    adata.var_names_make_unique()
    if "in_tissue" in adata.obs:
        adata = adata[adata.obs["in_tissue"] == 1].copy()

    counts = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)

    if label_file is None:
        return counts, coords, None

    import pandas as pd

    table = pd.read_csv(label_file, index_col=0).iloc[:, 0].dropna()
    shared = adata.obs_names.intersection(table.index)
    idx = adata.obs_names.get_indexer(shared)
    labels = pd.Categorical(table.loc[shared].astype(str)).codes
    return counts[idx], coords[idx], labels.astype(np.int64)
