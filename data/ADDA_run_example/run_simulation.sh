#!/bin/bash

# user inputs
N_orientations=150
Pcs=0.1
Pcc=0.1
Vratio=3.0
wvl=0.550
coating_RI_imag=0.00

# compile
make

# coat the cluster and extract Req
cp cluster.xyz particle
./shape_file <<< "${Pcs} ${Pcc} ${Vratio}"
coated_Req=$(awk -v "line=6" 'NR==line{ print $4 }' Cluster_Data.txt)

# make random orientations
python3 make_orientations.py --orientations=${N_orientations}

# run ADDA simulations
for ((i = 1; i <= N_orientations; i++))
do
    alpha=$(awk -v "line=${i}" 'NR==line{ print $0 }' alphas.txt)
    beta=$(awk -v "line=${i}" 'NR==line{ print $0 }' betas.txt)
    gamma=$(awk -v "line=${i}" 'NR==line{ print $0 }' gammas.txt)
    
    mpiexec ./adda_mpi -shape read coated_particle -eq_rad ${coated_Req} -lambda ${wvl} -orient ${alpha} ${beta} ${gamma} -dir orientation_${i} -m 1.95 0.79 1.6 ${coating_RI_imag}    </dev/null

done

rm particle
rm shape_file
rm shape_file.o


