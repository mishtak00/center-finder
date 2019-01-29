#!/bin/bash
#SBATCH --partition=standard --output=output_%a.txt -c 1 -t 10:00:00 --mem-per-cpu=50gb
#SBATCH -a 0-5
python test_fine_bins.py
