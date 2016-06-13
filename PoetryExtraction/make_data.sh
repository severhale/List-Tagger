#!/bin/bash
rm -f joined_data
LANG=en_EN sort classification > classification_tmp
LANG=en_EN sort training_data > training_data_tmp
LANG=en_EN join classification_tmp training_data_tmp > joined_data_tmp
i=0
while read -r word1 word2
do
i=$((i+1))
echo "$word2 # $word1" >> joined_data
done < joined_data_tmp
rm joined_data_tmp
rm classification_tmp
rm training_data_tmp
echo $i