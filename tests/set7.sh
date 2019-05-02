#!/bin/bash
#SBATCH --partition=standard --output=output_%a.txt -c 1 -t 2-10:00:00 --mem-per-cpu=100gb
#SBATCH -a 0-30
python test_scan.py

