#!/bin/sh

nthreads=(1 2 4 8 16 32)
mkdir measurements

echo "Performance measurements will be dumped to folder 'measurements'!!!"


for i in "${nthreads[@]}"
do
    export OMP_NUM_THREADS=$i
    echo $OMP_NUM_THREADS
    ./ssa 2> measurements/ssa_$i.txt

done
