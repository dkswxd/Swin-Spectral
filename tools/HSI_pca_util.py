import numpy as np
np.set_printoptions(threshold=np.inf)
import os
import tqdm
import cv2
import glob
import random

def main():
    #################################################################
    # settings
    force_recompute = False
    test_pac_vector = True
    bands_pca_show = 8
    dataset_root = './data/HSI/'
    hdr_folder = os.path.join(dataset_root, 'hdr_dir')
    mean_vector_path = os.path.join(dataset_root, 'mean_vector.npy')
    std_vector_path = os.path.join(dataset_root, 'std_vector.npy')
    correlation_mat_path = os.path.join(dataset_root, 'correlation_mat.npy')
    correlation_mat_heatmap_path = os.path.join(dataset_root, 'correlation_mat.png')
    pca_value_path = os.path.join(dataset_root, 'pca_value.npy')
    pca_vector_path = os.path.join(dataset_root, 'pca_vector.npy')
    pca_mean_vector_path = os.path.join(dataset_root, 'pca_mean_vector.npy')
    pca_std_vector_path = os.path.join(dataset_root, 'pca_std_vector.npy')
    pca_select_channel_path = os.path.join(dataset_root, 'pca_select_channel.npy')
    pca_select_channel_heatmap_path = os.path.join(dataset_root, 'pca_select_channel.png')
    test_pac_vector_path = os.path.join(dataset_root, "pca_test_{}.png")

    ENVI_data_type = [None,np.uint8,np.int16,np.int32,np.float32,np.float64,
                      None,None,None,None,None,None,np.uint16,np.uint32,]
    #################################################################
    # sample one hdr file, assuming all file has same shape
    print('sample one hdr file, assuming all files have same shape')
    filename = glob.glob(os.path.join(hdr_folder, '*.hdr'))[0]
    count = len(glob.glob(os.path.join(hdr_folder, '*.raw')))
    print(f'filename: {filename}')
    hdr = dict()
    with open(filename) as f:
        for line in f.readlines():
            if '=' in line:
                key, value = line.split('=')
                hdr[key.strip()] = value.strip()
    bands, height, width, dtype = int(hdr['bands']), int(hdr['lines']), int(hdr['samples']), int(hdr['data type'])
    print(f'bands: {bands}, height: {height}, width: {width}')
    #################################################################
    # calculate mean and std
    if not os.path.exists(mean_vector_path) or force_recompute:
        print('calculating mean')
        mean_vector = np.zeros((bands,), dtype=np.float32)
        for filename in tqdm.tqdm(glob.glob(os.path.join(hdr_folder, '*.raw'))):
            raw = np.fromfile(filename, dtype=ENVI_data_type[dtype])
            raw = raw.reshape((bands, height, width))
            mean_vector += np.mean(raw, axis=(1, 2))
        mean_vector /= count
        np.save(mean_vector_path, mean_vector)
    else:
        print(f'using mean in {mean_vector_path}')
        mean_vector = np.load(mean_vector_path)
    # print('mean = \n', mean_vector)

    mean_vector = mean_vector.reshape((bands, 1, 1))
    std_vector = np.zeros((bands,), dtype=np.float32)
    if not os.path.exists(std_vector_path) or force_recompute:
        print('calculating mean')
        for filename in tqdm.tqdm(glob.glob(os.path.join(hdr_folder, '*.raw'))):
            raw = np.fromfile(filename, dtype=ENVI_data_type[dtype])
            raw = raw.reshape((bands, height, width)).astype(np.float32)
            raw -= mean_vector
            std_vector += np.mean(np.power(raw, 2), axis=(1, 2))
        std_vector /= count
        std_vector = np.sqrt(std_vector)
        np.save(std_vector_path, std_vector)
    else:
        print(f'using std in {std_vector_path}')
        std_vector = np.load(std_vector_path)
    # print('std = \n', std_vector)

    #################################################################
    # calculate correlation mat
    mean_vector = np.reshape(mean_vector, (1, -1))
    std_vector = np.reshape(std_vector, (1, -1))
    if not os.path.exists(correlation_mat_path) or force_recompute:
        print('calculating correlation mat')
        correlation_mat = np.zeros((bands, bands),dtype=np.float32)
        for filename in tqdm.tqdm(glob.glob(os.path.join(hdr_folder, '*.raw'))):
            raw = np.fromfile(filename, dtype=ENVI_data_type[dtype])
            raw = raw.reshape((bands, height, width))
            data = np.transpose(raw, (1, 2, 0))
            data = np.reshape(data, (-1, bands)).astype(np.float32)
            data -= mean_vector
            data /= std_vector
            correlation_mat += np.dot(data.T, data) / (height * width)
        correlation_mat /= count
        np.save(correlation_mat_path, correlation_mat)
    else:
        print(f'using correlation_mat in {correlation_mat_path}')
        correlation_mat = np.load(correlation_mat_path)
    correlation_mat_heatmap = cv2.resize((correlation_mat * 254).astype(np.uint8),
                                         (bands * 8, bands * 8), interpolation=cv2.INTER_NEAREST)
    correlation_mat_heatmap = cv2.applyColorMap(correlation_mat_heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(correlation_mat_heatmap_path, correlation_mat_heatmap)
    # print('correlation_mat = \n', correlation_mat)

    #################################################################
    # calculate pca vector
    if not os.path.exists(pca_vector_path) or not os.path.exists(pca_value_path) or force_recompute:
        print('calculating pca vector')
        pca_value, pca_vector = np.linalg.eigh(correlation_mat)
        pca_value = pca_value[::-1]
        pca_vector = pca_vector[:,::-1]
        np.save(pca_value_path, pca_value)
        np.save(pca_vector_path, pca_vector)
    else:
        print(f'using pca_vector in {pca_vector_path}')
        pca_value = np.load(pca_value_path)
        pca_vector = np.load(pca_vector_path)
    # print('pca_value=\n', pca_value)
    # print('pca_vector=\n', pca_vector)


    #################################################################
    # test pca vector
    if test_pac_vector:
        filename = glob.glob(os.path.join(hdr_folder, '*.raw'))[random.randint(0, count - 1)]
        print(f'testing pca vector with filename: {filename}')
        raw = np.fromfile(filename, dtype=ENVI_data_type[dtype])
        raw = raw.reshape((bands, height, width))
        data = np.transpose(raw, (1, 2, 0))
        data = np.reshape(data, (-1, bands)).astype(np.float32)
        data -= mean_vector
        data /= std_vector
        data_pca = np.dot(data, pca_vector[:,:bands_pca_show])
        data_pca = np.reshape(data_pca, (height, width, bands_pca_show))
        for i in range(bands_pca_show):
            cv2.imwrite(test_pac_vector_path.format(i), (data_pca[:,:,i] * 4 + 128).astype(np.uint8))

    #################################################################
    # calculate pca mean and std
    if not os.path.exists(pca_mean_vector_path) or force_recompute:
        print('calculating pca mean')
        pca_mean_vector = np.zeros((bands,), dtype=np.float32)
        for filename in tqdm.tqdm(glob.glob(os.path.join(hdr_folder, '*.raw'))):
            raw = np.fromfile(filename, dtype=ENVI_data_type[dtype])
            raw = raw.reshape((bands, height, width))
            data = np.transpose(raw, (1, 2, 0))
            data = np.reshape(data, (-1, bands)).astype(np.float32)
            data -= mean_vector
            data /= std_vector
            data_pca = np.dot(data, pca_vector)
            pca_mean_vector += np.mean(data_pca, axis=0)
        pca_mean_vector /= count
        np.save(pca_mean_vector_path, pca_mean_vector)
    else:
        print(f'using mean in {pca_mean_vector_path}')
        pca_mean_vector = np.load(pca_mean_vector_path)
    # print('mean = \n', mean_vector)

    pca_mean_vector = pca_mean_vector.reshape((1, bands))
    pca_std_vector = np.zeros((bands,), dtype=np.float32)
    if not os.path.exists(pca_std_vector_path) or force_recompute:
        print('calculating pca std')
        for filename in tqdm.tqdm(glob.glob(os.path.join(hdr_folder, '*.raw'))):
            raw = np.fromfile(filename, dtype=ENVI_data_type[dtype])
            raw = raw.reshape((bands, height, width))
            data = np.transpose(raw, (1, 2, 0))
            data = np.reshape(data, (-1, bands)).astype(np.float32)
            data -= mean_vector
            data /= std_vector
            data_pca = np.dot(data, pca_vector)
            data_pca -= pca_mean_vector
            pca_std_vector += np.mean(np.power(data_pca, 2), axis=0)
        pca_std_vector /= count
        pca_std_vector = np.sqrt(pca_std_vector)
        np.save(pca_std_vector_path, pca_std_vector)
    else:
        print(f'using std in {pca_std_vector_path}')
        pca_std_vector = np.load(pca_std_vector_path)
    # print('std = \n', std_vector)

    #################################################################
    # calculate pca_select_channel
    if not os.path.exists(pca_select_channel_path) or force_recompute:
        print(f'calculating pca_select_channel_correlation_mat')
        pca_select_channel_correlation_mat = np.zeros((bands, bands), dtype=np.float32)
        for filename in tqdm.tqdm(glob.glob(os.path.join(hdr_folder, '*.raw'))):
            raw = np.fromfile(filename, dtype=ENVI_data_type[dtype])
            raw = raw.reshape((bands, height, width))
            data = np.transpose(raw, (1, 2, 0))
            data = np.reshape(data, (-1, bands)).astype(np.float32)
            data -= mean_vector
            data /= std_vector
            data_pca = np.dot(data, pca_vector)
            data_pca -= pca_mean_vector
            data_pca /= pca_std_vector
            pca_select_channel_correlation_mat += np.dot(data_pca.T, data) / (height * width)
        pca_select_channel_correlation_mat /= count
        np.save(pca_select_channel_path, pca_select_channel_correlation_mat)
    else:
        print(f'using pca_select_channel_correlation_mat in {pca_select_channel_path}')
        pca_select_channel_correlation_mat = np.load(pca_select_channel_path)
    print('pca_select_channel_correlation_mat result:')
    print(np.argmax(np.abs(pca_select_channel_correlation_mat), axis=1))
    pca_select_channel_heatmap = cv2.resize((np.abs(pca_select_channel_correlation_mat) * 254).astype(np.uint8),
                                            (bands * 8, bands * 8), interpolation=cv2.INTER_NEAREST)
    pca_select_channel_heatmap = cv2.applyColorMap(pca_select_channel_heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(pca_select_channel_heatmap_path, pca_select_channel_heatmap)

    #################################################################
    # calculate correlation_select_channel
    print(f'calculating correlation_select_channel')
    C1, C2, C3 = 0, 0, 0
    for c1 in range(bands):
        for c2 in range(c1, bands):
            for c3 in range(c2, bands):
                if correlation_mat[c1][c2] + correlation_mat[c1][c3] + correlation_mat[c2][c3] < \
                   correlation_mat[C1][C2] + correlation_mat[C1][C2] + correlation_mat[C2][C3]:
                    C1, C2, C3 = c1, c2, c3
    print(f'correlation_select_channel result: {C1}, {C2}, {C3}')

if __name__ == '__main__':
    main()























