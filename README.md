# code-essentials

Most commonly used Julia functions by me. Short discriptions of module files in the repo:

| File              | Description                                                                                  |
|-------------------|----------------------------------------------------------------------------------------------|
| CommonImports  | Commonly used libraries and dependencies.                                                   |
| DataStruct     | Define core data structures for numeric workflows.                                          |
| BaseIO         | Handle basic input/output routines (e.g., file reads and writes). Centralized for consistency. |
| ArrayFunctions | Perform array-based computations and transformations (e.g., meshgridding, sub-gridding, array rotations, appending ghost cells). |
| TurbAnalysis    | Functions for turbulence simulation analysis (e.g., getting Ma, Ms, energy spectra, energy time-evolution). |
| DecompMHD      | Implementing MHD mode decomposition.                                                         |
| SPAFunctions   | Functions used for numerical tests of the SPA++ method from [*Pavaskar et al. 2024*](doi.org/10.3847/1538-4357/ad5af5). |
| ShockFind      | Shockfind algorithm similar to [*Lehmann et al. 2016*](doi.org/10.1093/mnras/stw2015).             |
| tmpdec         | Decomposition tests and code from Kirit.                                                    |
