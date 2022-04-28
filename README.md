# Shell_MDO_ROM (ShMORe?)
Reduced-Order Modeling for shell analysis with PENGoLINS for MDO applications

This repository requires [PENGoLINS](https://github.com/hanzhao2020/PENGoLINS) and its dependencies.

To run the main script in parallel (with mpi), use a terminal to run:
```
mpirun -n X python3 evtol_wing.py
```
Where `X` is the desired number of parallel processes.
