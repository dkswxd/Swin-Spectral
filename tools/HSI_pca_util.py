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
    bands_pca_show = 8
    dataset_root = './data/HSI/'
    hdr_folder = os.path.join(dataset_root, 'hdr_dir')
    pca_vector_path = os.path.join(dataset_root, 'pca_vector.npy')
    mean_vector_path = os.path.join(dataset_root, 'mean_vector.npy')
    std_vector_path = os.path.join(dataset_root, 'std_vector.npy')

    ENVI_data_type = [None,np.uint8,np.int16,np.int32,np.float32,np.float64,
                      None,None,None,None,None,None,np.uint16,np.uint32,]
    #################################################################
    # sample one hdr file, assuming all file has same shape
    print('sample one hdr file, assuming all file has same shape')
    filename = glob.glob(os.path.join(hdr_folder, '*.hdr'))[0]
    print(f'filename: {filename}')
    hdr = dict()
    with open(filename) as f:
        for line in f.readlines():
            if '=' not in line:
                continue
            else:
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                hdr[key] = value
    bands = int(hdr['bands'])
    height = int(hdr['lines'])
    width = int(hdr['samples'])
    dtype = int(hdr['data type'])
    print(f'bands: {bands}, height: {height}, width: {width}')
    #################################################################
    # get mean and std
    print('get mean and std')
    mean_vector = np.zeros((bands),dtype=np.float32)
    std_vector = np.zeros((bands),dtype=np.float32)
    count = 0

    for filename in tqdm.tqdm(glob.glob(os.path.join(hdr_folder, '*.raw'))):
        raw = np.fromfile(filename, dtype=ENVI_data_type[dtype])
        raw = raw.reshape((bands, height, width))
        mean_vector += np.mean(raw, axis=(1, 2))
        std_vector += np.std(raw, axis=(1, 2))
        count += 1

    mean_vector /= count
    std_vector /= count
    print('mean = \n',mean_vector)
    np.save(mean_vector_path, mean_vector)
    print('std = \n',std_vector)
    np.save(std_vector_path, std_vector)

    #################################################################
    # get cov mat
    print('get cov mat')
    mean_vector = np.reshape(mean_vector, (1, -1))
    std_vector = np.reshape(std_vector, (1, -1))

    cov_mat = np.zeros((bands, bands),dtype=np.float32)
    for filename in tqdm.tqdm(glob.glob(os.path.join(hdr_folder, '*.raw'))):
        raw = np.fromfile(filename, dtype=ENVI_data_type[dtype])
        raw = raw.reshape((bands, height, width))
        data = np.transpose(raw, (1, 2, 0))
        data = np.reshape(data, (-1, bands)).astype(np.float32)
        data -= mean_vector
        data /= std_vector
        cov_mat += np.dot(data.T, data) / (height * width)

    cov_mat /= count

    #################################################################
    # get pca vector
    print('get pca vector')
    pca_value, pca_vector = np.linalg.eigh(cov_mat)
    pca_value = pca_value[::-1]
    pca_vector = pca_vector[:,::-1]
    print('pca_value=\n', pca_value)
    print('pca_vector=\n', pca_vector)
    np.save(pca_vector_path, pca_vector)

    #################################################################
    # test pca vector
    print('test pca vector')
    filename = glob.glob(os.path.join(hdr_folder, '*.raw'))[random.randint(0, count - 1)]
    print(f'filename: {filename}')
    raw = np.fromfile(filename, dtype=ENVI_data_type[dtype])
    raw = raw.reshape((bands, height, width))
    data = np.transpose(raw, (1, 2, 0))
    data = np.reshape(data, (-1, bands)).astype(np.float32)
    data -= mean_vector
    data /= std_vector
    data = np.dot(data, pca_vector[:,:bands_pca_show])
    data = np.reshape(data, (height, width, bands_pca_show))
    for i in range(bands_pca_show):
        cv2.imwrite(f"pca_test_{i}.png", (data[:,:,i] * 4 + 128).astype(np.uint8),)

if __name__ == '__main__':
    main()























