#!/bin/bash

# Iterate through the list and call the python script with each element
for element in "03R" "07R" "09R" ;
do
    python3 unconstrained_pipeline.py device=$element
done