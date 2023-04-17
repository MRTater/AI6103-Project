#!/bin/bash

# Remove error and output files
rm -f *.err *.out

# Remove files in the result directory
rm -f result/*

# Remove files in the model directory
rm -f model/*

echo "Cleaned up files!"