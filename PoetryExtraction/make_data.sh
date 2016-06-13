#!/bin/bash
rm joined_data
join classification training_data > joined_data_tmp
while read -r word1 word2
do
echo "$word2 # $word1" >> joined_data
done < joined_data_tmp
rm joined_data_tmp