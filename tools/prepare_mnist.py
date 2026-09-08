#!/usr/bin/env python3
"""Explicit MNIST download and deterministic train/validation preparation; stdlib only."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
from urllib.request import urlopen

BASE = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
SOURCE = 'https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py'
FILES = {
    'train-images-idx3-ubyte': ('f68b3c2dcbeaaa9fbdd348bbdeb94873', 60000 * 784 + 16),
    'train-labels-idx1-ubyte': ('d53e105ee54ea40749a09fcbcd1e9432', 60000 + 8),
    't10k-images-idx3-ubyte': ('9fb629c4189551a2d022fa330f9573f3', 10000 * 784 + 16),
    't10k-labels-idx1-ubyte': ('ec29112dd5afa0611ce80d1b7f02629c', 10000 + 8),
}

def store(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'{path} exists with different contents; move it aside first')
        return
    temporary = path.with_name(path.name + '.partial')
    temporary.write_bytes(data)
    temporary.replace(path)

def parse(images, labels):
    if len(images) < 16 or len(labels) < 8:
        raise ValueError('Truncated IDX header')
    magic, n, rows, cols = struct.unpack('>4I', images[:16])
    lmagic, ln = struct.unpack('>2I', labels[:8])
    if magic != 2051 or lmagic != 2049 or n != ln or min(n, rows, cols) == 0:
        raise ValueError('Invalid IDX dimensions, magic, or matching counts')
    if len(images) != 16 + n * rows * cols or len(labels) != 8 + n:
        raise ValueError('IDX payload length mismatch')
    if any(label > 9 for label in labels[8:]):
        raise ValueError('MNIST labels must be 0 through 9')
    return n, rows, cols

def split(images, labels, per_class):
    n, rows, cols = parse(images, labels)
    held = set()
    for label in range(10):
        indices = [i for i in range(n) if labels[8+i] == label]
        if len(indices) <= per_class:
            raise ValueError('Each class needs more rows than its validation allocation')
        # Hash order is independent of Python random-generator versions.
        indices.sort(key=lambda i: hashlib.sha256(f'first-principles-v1:{i}'.encode()).digest())
        held.update(indices[:per_class])
    result = {}
    width = rows * cols
    for name, indices in [('fit', [i for i in range(n) if i not in held]), ('validation', sorted(held))]:
        result[name+'-images-idx3-ubyte'] = struct.pack('>4I', 2051, len(indices), rows, cols) + b''.join(images[16+i*width:16+(i+1)*width] for i in indices)
        result[name+'-labels-idx1-ubyte'] = struct.pack('>2I', 2049, len(indices)) + bytes(labels[8+i] for i in indices)
    return result, sorted(held)

def self_test():
    images = struct.pack('>4I', 2051, 30, 1, 1) + bytes(range(30))
    labels = struct.pack('>2I', 2049, 30) + bytes(i % 10 for i in range(30))
    parts, held = split(images, labels, 1)
    assert len(held) == 10 and len(set(held)) == 10
    assert split(images, labels, 1) == (parts, held)
    fit = set(parts['fit-images-idx3-ubyte'][16:])
    validation = set(parts['validation-images-idx3-ubyte'][16:])
    assert not fit & validation and fit | validation == set(range(30))
    assert sorted(parts['validation-labels-idx1-ubyte'][8:]) == list(range(10))
    try:
        parse(images[:-1], labels)
    except ValueError:
        pass
    else:
        raise AssertionError('Truncated data accepted')
    print('Preparation self-check passed; no network used.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Download about 12 MB and prepare 55k/5k splits')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--directory', type=Path, default=Path('datasets/downloads/mnist'))
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if not args.download:
        if not args.self_test:
            parser.print_help()
        return
    args.directory.mkdir(parents=True, exist_ok=True)
    provenance = {'dataset': 'MNIST', 'authors': 'Yann LeCun, Corinna Cortes, Christopher J. C. Burges', 'reference': SOURCE, 'mirror': BASE, 'split': 'Original training only: 500 examples per class selected by SHA-256 order; official test unchanged.', 'files': {}}
    raw = {}
    for name, (md5, size) in FILES.items():
        archive = args.directory / (name+'.gz')
        if archive.exists():
            data = archive.read_bytes()
        else:
            print('Downloading', name+'.gz', flush=True)
            with urlopen(BASE+name+'.gz', timeout=45) as response:
                data = response.read(20_000_001)
        if len(data) > 20_000_000 or hashlib.md5(data, usedforsecurity=False).hexdigest() != md5:
            raise ValueError(f'Archive checksum mismatch: {name}; no extraction performed')
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as compressed:
            decoded = compressed.read(size+1)
        if len(decoded) != size:
            raise ValueError(f'Unexpected decoded size for {name}')
        store(archive, data)
        store(args.directory/name, decoded)
        raw[name] = decoded
        provenance['files'][name] = {'url': BASE+name+'.gz', 'archive_md5': md5, 'sha256': hashlib.sha256(decoded).hexdigest(), 'bytes': size}
    parts, held = split(raw['train-images-idx3-ubyte'], raw['train-labels-idx1-ubyte'], 500)
    parse(raw['t10k-images-idx3-ubyte'], raw['t10k-labels-idx1-ubyte'])
    for name, data in parts.items():
        store(args.directory/name, data)
        provenance['files'][name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    provenance['validation_original_indices'] = held
    store(args.directory/'provenance.json', (json.dumps(provenance, indent=2)+'\n').encode())
    print('Prepared 55,000 fit / 5,000 validation / 10,000 untouched test examples in', args.directory)

if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
