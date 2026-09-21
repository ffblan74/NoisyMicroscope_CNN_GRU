"""Prepare balanced classroom subsets from the official BloodMNIST partitions."""
from pathlib import Path
import hashlib
import ssl
import urllib.request
import certifi
import numpy as np

ROOT = Path(__file__).resolve().parent
URL = 'https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1'
MD5 = '7053d0359d879ad8a5505303e11de1dc'

def main():
    directory = ROOT / 'data'
    directory.mkdir(exist_ok=True)
    output = directory / 'blood_mnist_subset.npz'
    if output.exists():
        print('Keeping existing subset:', output)
        return
    raw = directory / 'bloodmnist.npz'
    if not raw.exists():
        print('Downloading official BloodMNIST...', flush=True)
        with urllib.request.urlopen(URL, context=ssl.create_default_context(cafile=certifi.where()), timeout=120) as response:
            content = response.read()
        if hashlib.md5(content).hexdigest() != MD5:
            raise ValueError('Official checksum mismatch')
        with raw.open('xb') as handle:
            handle.write(content)
    if hashlib.md5(raw.read_bytes()).hexdigest() != MD5:
        raise ValueError('Invalid source cache')
    rng = np.random.default_rng(42)
    result = {}
    with np.load(raw, allow_pickle=False) as source:
        for split, per_class in [('train', 400), ('val', 80), ('test', 100)]:
            labels = source[split + '_labels'].reshape(-1)
            selected = []
            for label in range(8):
                ids = np.flatnonzero(labels == label)
                if len(ids) < per_class:
                    raise ValueError(f'Insufficient images for {split}, class {label}')
                selected.extend(rng.permutation(ids)[:per_class])
            selected = rng.permutation(selected)
            result[split + '_images'] = source[split + '_images'][selected]
            result[split + '_labels'] = labels[selected]
    with output.open('xb') as handle:
        np.savez_compressed(handle, **result)
    print('Prepared 3200 train / 640 validation / 800 test RGB images:', output)

if __name__ == '__main__':
    main()
