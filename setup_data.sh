# Small script to download the data for the notebooks 
# Run this script to download some sample into ./data/
mkdir -p data
rm -rf data/*
curl -o data/dataset.zip https://zenodo.org/records/13305156/files/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip?download=1?
unzip -o data/dataset.zip -d data
curl -o data/dataset.zip https://zenodo.org/records/13305316/files/20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip?download=1?
unzip -o data/dataset.zip -d data   
rm data/dataset.zip