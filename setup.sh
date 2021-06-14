#!/bin/bash

mkdir -p data/kinetics/scenescores
mkdir -p data/kinetics/objectscores
wget -r -np -nH --cut-dirs=5 -P data/kinetics/scenescores/ -R "index.html*" https://isis-data.science.uva.nl/mettes/zero-shot-actions/data/kinetics/scene-avg
wget -r -np -nH --cut-dirs=5 -P data/kinetics/objectscores/ -R "index.html*" https://isis-data.science.uva.nl/mettes/zero-shot-actions/data/kinetics/global-avg
cd auxiliaryscripts
bash renameavgfiles.sh
cd ..
python getfasttextembs.py
python getbertembs.py
