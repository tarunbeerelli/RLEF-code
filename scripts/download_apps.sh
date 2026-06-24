#!/bin/bash
# Download APPS dataset (~1.3GB)
# Run once after cloning the repo on the cloud machine
set -e

echo "Downloading APPS dataset..."
mkdir -p data/raw

python3 -c "
import urllib.request, tarfile, pathlib

url  = 'https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz'
dest = pathlib.Path('data/raw/APPS.tar.gz')
dest.parent.mkdir(parents=True, exist_ok=True)

print('Downloading (~1.3GB)...')
urllib.request.urlretrieve(url, dest,
    reporthook=lambda b, bs, t: print(f'{min(b*bs,t)/1e6:.0f}MB / {t/1e6:.0f}MB', end='\r'))

print()
print('Extracting...')
with tarfile.open(dest) as tar:
    tar.extractall('data/raw/')

dest.unlink()  # remove tarball after extraction
print('Done. APPS dataset ready at data/raw/APPS/')
"
