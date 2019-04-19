#!/bin/bash
#SBATCH --partition=standard --output=output_%a.txt -c 1 -t 1-5:00:00 --mem-per-cpu=100gb
#SBATCH -a 0-10
python test_scan.py

