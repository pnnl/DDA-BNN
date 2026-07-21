## Quick start

This folder shows an example ADDA run that can be used to run new ADDA simulations. It includes an example cluster xyz file, discretization and coating alogorithm, and ADDA command line. An example ADDA can be run following:

1. Download the adda source code from https://github.com/adda-team/adda.git
2. Compile the MPI version of ADDA
3. Ensure that `adda_mpi` file is in working directory
4. Run `./run_simulation.sh`. This file will discretize the particle defined in cluster.xyz, coat, and run ADDA simulation

User options for the coating RI, amount of coating, wavelength, and number of orientations can be found in `run_simulation.sh`. 
