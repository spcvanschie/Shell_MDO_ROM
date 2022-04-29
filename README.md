# Shell_MDO_ROM (ShMORe?): Reduced-Order Modeling for shell analysis with PENGoLINS for MDO applications
### Dependencies

This repository requires [PENGoLINS](https://github.com/hanzhao2020/PENGoLINS) and its dependencies.

### Parallel running

To run the main script in parallel (with mpi), use a terminal to run:
```
mpirun -n X python3 evtol_wing.py
```
Where `X` is the desired number of parallel processes.

### Visualization of results

The displacements and stress fields can be visualized in Paraview by going to the View tab and ticking the "Python Shell" box. This adds a "Run Script" option in the bottom right corner of the screen, which can be used to load and run `view_results.py`. 
If errors are being thrown while executing `view_results.py`, check the default value of the `file_path` input argument in `view_results.py` and change it if necessary.
