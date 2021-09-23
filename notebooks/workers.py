from localedb.clustering import dist_mat

def dist_map_worker(a, metric='cwt-spca', wavelet='cmor'):
    return dist_mat(a, metric, wavelet=wavelet)
