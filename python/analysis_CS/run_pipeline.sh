#!/bin/bash

for i in "02L" "02R" "03L" "03R" "07L" "07R" "09L" "09R" "16L" "16R"; do
    python3 ./pipeline/pipeline_main.py setup.device=$i
done

for i in "02L" "02R" "03L" "03R" "07L" "07R" "09L" "09R" "16L" "16R"; do
    python3 ./pipeline/pipeline_main.py setup.device=$i
done

for i in "02L" "02R" "03L" "03R" "07L" "07R" "09L" "09R" "16L" "16R"; do
    python3 ./pipeline/pipeline_main.py setup.device=$i
done