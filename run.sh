#!/bin/bash
echo 'Downloading dataset ...'
curl https://archive.ics.uci.edu/ml/machine-learning-databases/00365/Polish_companies_bankruptcy_data.zip -o data.zip
echo 'Unzipping data ...'
unzip -e data.zip
echo '-- All Setup done'
echo '-----------------'
echo 'Running Script:'
python bankruptcy.py
echo 'All done. Classifier classes dumps available in model_dumps folder.'
