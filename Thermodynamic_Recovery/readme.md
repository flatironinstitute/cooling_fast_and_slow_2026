# Code for Thermodynamic Inference

This folder contains all code and data required to reproduce the thermodynamic inference results in the manuscript, in particular Figures 6, S12, S13, S14, and S15.

For anyone looking to use the thermodynamic inference procedure for their own data, we do not recommend trying to adapt these scripts, since they were built bespoke for the data analyzed in this manuscript and are not easily modifiable. We have instead developed a separate python package which can be used out of the box on general datasets. This package can be found at the following link: https://github.com/Matthew-Leighton/Cooling_Thermodynamic_Recovery.

For the 6-state toy model, we use data from Langevin dynamics simulations. The dataset is large enough that uncertainties on cooled state probabilities are negligible, significantly simplifying the analysis.

Using the Trp-cage data from extensive MD simulations, we ran 3 different thermodynamic recovery experiments. First using all of the cooling simulation data (see folder Inference_AllCoolings), then using only the three slowest cooling rates (folder Inference_ThreeSlowestCoolings), and finally using only the slowest cooling rate (folder Inference_OnlySlowestCooling).

# Inference
Each folder contains a script labeled "Thermodynamic_Inference.ipynb" which takes in cooled state populations, and generates sets of equilibrium state probabilities and microstate probabilities consistent with the thermodynamic constraints. These only differ between folders in which cooled distributions they use for inference. Note that running the inference scripts to produce a comparable number of samples to that used in our manuscript (the default settings) will take several days on a standard laptop.

For the 6-state toy model, there are few enough states (and no uncertainties) that the distribution space can be explored systematically to determine the boundaries of the thermdynamically allowed region. This inference script is much faster than those for the Trp-cage data.

# Plotting
Figure X is plotted using a script Plot_Figure_X.ipynb. Each plotting figure takes in data from the inference script outputs, and uses it to plot the figures in the main paper. Since the data is contained within the folders already, all figures can be reproduced simply by running these python notebooks.