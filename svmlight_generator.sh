#!/bin/bash
join classification training_data > joined_data_tmp
cut -d' ' -f2- joined_data_tmp > joined_data
rm joined_data_tmp