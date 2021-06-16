#!/bin/bash

# path of parent folder
path=../data/kinetics/objectscores

# for each folder in parent folder
for folder in "$path"/*; do

  # get folder name
  foldername=${folder##*/}

  # for each subfolder in folder
  for subfolder in "$path"/"$foldername"/*; do

    # get subfolder name
    subfoldername=${subfolder##*/}

    # rename path/folder/subfoldername to path/folder/folder_filename
    mv "$path"/"$foldername"/"$subfoldername"/avg.npy "$path"/"$foldername"/"$subfoldername"/avg-features.npy

  done

  echo $foldername files renamed!

done

echo finished!
