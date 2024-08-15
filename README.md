# npx-analysis

This repository contains the code used for the analysis contained in Palacios et al., bioRxiv, 2024, [Golgi cells regulate timing and variability of information transfer in the cerebellar-behavioural loop]

[Golgi cells regulate timing and variability of information transfer in the cerebellar-behavioural loop]:https://www.biorxiv.org/content/10.1101/2024.07.10.602852v1.full.pdf

# Organisation
There are four basic modules: the first two, load_npx.py and whisk_analysis.py, are used to pre-process npx and video data, respectively; the former works on the output of [Kilosort 2] and [Phy2], the latter uses the output of [DeepLabCut] to derive whisking periods and properties (e.g., angle) from whisker landmarks. The third module, meta_data.py, contains metadata such as the paths to npx and video data as well as other information necessary for analysis. The fourth module, align_fun.py, is responsible for aligning neuronal and whisking data, and its functions are used in other modules.

Brief description of other modulels:
* raster_plot.py: generate raster plots from one whisking event for one recording.
* peth_module.py: generate the peri-event time histogram for one recording.
* peth_compare.py: generate the peri-event time histogram from pre- and post-drug application for one recording.
* lowdim_visualisation_old.py: run and analyse output of PCA for one recording.
* glm_module_decoding.py: decode whisking activity from PCA data for one recording.
* tc_module.py: compute tuning curves and their properties for both control data and for control vs GlyT2+CNO data.
* tc_fit.py: fit b-spline model to raw tuning curves.
* tc_module_hpc.py: fit b-spline model to pre- and post-drop raw tuning curves; was run on high-performace computer, so argument (recording number) needs to be specified from the terminal.
* fit_spkcnt.py module: analyse difference between experimental conditions based on every recording's total spike counts.
* whisk_plot.py: compute whisking properties and their change pre- vs post-drop in contol vs GlyT2+CNO data

[Kilosort 2]:https://github.com/MouseLand/Kilosort/tree/kilosort2
[Phy2]:https://github.com/cortex-lab/phy
[DeepLabCut]:http://www.mackenziemathislab.org/deeplabcut

# Licence
Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg



