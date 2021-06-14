#!/bin/bash

# path of parent folder
path=~/Documents/Uni/_T/main/data/ucf-101/scenescores-resnet50

# for each folder in parent folder
for folder in "$path"/*; do

  # get folder name
  foldername=${folder##*/}

  # Set comma as delimiter
  IFS='_' read -a strarr <<< "$foldername"
  mkdir -p "$path"/"${strarr[1]}"

  mv "$path"/"$foldername" "$path"/"${strarr[1]}"/"$foldername"

  echo $foldername folder moved!

done
echo finished!
