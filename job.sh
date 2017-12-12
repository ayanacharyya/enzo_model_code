#!/bin/bash
#PBS -P ek9
#PBS -q express
#PBS -l walltime=5:00:00
#PBS -l ncpus=128
#PBS -l mem=256GB
#PBS -l wd
#PBS -N run_DD0600_lgf_exp240000_realsn1
#PBS -j oe
#PBS -m bea
#PBS -M u5877042@anu.edu.au

python call_loopcall_server.py 1>shell.out 2>&1
