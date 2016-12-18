This repository contains codes, data, and crystal structure files to reproduce the results in the following article:

> C. Simon, E. Braun, C. Carraro, B. Smit. Statistical mechanical model of gas adsorption in porous crystals with dynamic moieties. *Proceedings of the National Academy of Sciences*. (2016)

All codes written in the [Julia](http://julialang.org/) programming language.

* `utils.jl`: contains data structures and constants that are shared with modules for computing exact and mean field theory solutions as well as for Monte Carlo simulations.
* `exact_soln.jl`: contains functions to produce exact solution.
* `higher_dim_gcmc.jl`: contains functions to run Monte Carlo simulations in an arbitrary number of dimensions.
* `MFT_soln.jl`: contains functions to compute mean field solution.

### Interactive Fig. 1

* `Interact.ipynb` is a [Jupyter Notebook](http://jupyter.org/) that displays an interactive version of Fig. 1 in the main text.

### Reproducing plots in the main text and SI.

* `Rotating ligand model.ipynb`: reproduces most plots in the main text, loads modules in `*.jl` for computations.
* `Higher dimension Monte Carlo simulations.ipynb`: run 2D Monte Carlo simulations.
* `NVT.ipynb`: reproduce NVT simulations of 2D simulation for Fig. S13.

### Density functional theory calculations for MIL-91(Al) case study

The DFT directory contains VASP input files and the output relaxed crystal structure.
* `POSCAR.cif` : experimental crystal structure as provided by Guillaume Maurin from the study of Llewellyn PL, et al. (2015) Structural origin of unusual CO2 adsorption behavior of a small-pore aluminum bisphosphonate MOF. *J Phys Chem C* 119(8):4208-4216. This was converted to the VASP input file `POSCAR`.
* `CONTCAR.cif` : DFT-relaxed crystal structure as converted from the VASP output file `CONTCAR`.
* `INCAR`, `KPOINTS`, `POSCAR`, `POTCAR` : standard VASP input files.
* `CONTCAR` : standard VASP output file.
* `vdw_kernel.bindat` : precalculated kernel required for using the rev-vdW-DF2 functional.
