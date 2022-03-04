# SlurmCluster on JUWELS

## Singularity image

The below setup instructions prepare a container such that it can communicate with the host system wide slurm installation. 
Custom code execution is not affected.

 - The Singularity container to be used needs `ipykernel` to run with jupyter.
 - The container needs access to some host-system-wide libraries. 
   See the binds in the `kernel.json` and `jupyter_kernel_recipe.def` file. 
   This is JUWELS specific.
 - The host environment cannot be used, since some executable paths would be overwritten. 
   The container must be run with the `--cleanenv` option.
 