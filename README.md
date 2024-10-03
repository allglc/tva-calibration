# TvA calibration

## Simplified code for easy reuse

The notebook `main.ipynb` contains code to calibrate ImageNet models, display reliability diagrams, and study overfitting. You should probably start with this.

`calibrators.py` contains code for calibration methods. Implemented scaling methods are temperature scaling, vector scaling, and Dirichlet calibration, with and without TvA and regularization. Binary methods rely on the library netcal, and our code can use these methods with one-versus-all (the standard multiclass to binary reformulation) or top-versus-all (TvA).

`evaluation.py` contains code to compute ECE (equal-size or equal-mass bins), accuracy, average confidence, AUROC, and Brier score.


## Full code for transparency and reproducibility
We also provide the full code to reproduce all our experiments in the `full_code` folder. However, it will not run out of the box as the paths were anonymized and data files removed.

`full_code/benchmark_calibration.py` contains code to apply the different calibration methods, with and without TvA, and save many metrics. Results for image classification and text classification with pre-trained language models in the paper come from this script.

`full_code/benchmarking-uncertainty-estimation-performance-main/` contains functions to compute metrics (from [https://github.com/IdoGalil/benchmarking-uncertainty-estimation-performance](https://github.com/IdoGalil/benchmarking-uncertainty-estimation-performance)).

`full_code/focal_calibration/` contains pre-trained models for CIFAR (from [https://github.com/torrvision/focal_calibration](https://github.com/torrvision/focal_calibration)).

`full_code/imax_calib/` contains the I-Max baseline (from [https://github.com/boschresearch/imax-calibration](https://github.com/boschresearch/imax-calibration)).

`full_code/LinC-main` contains code of calibration for large language models using in-context learning (from [https://github.com/mominabbass/LinC](https://github.com/mominabbass/LinC)). We created the files `full_code/LinC-main/benchmark_tva.py` to compute and save metrics for applying HB_TvA on top of LinC (Table 11 in the paper) and `full_code/LinC-main/analyse_results.ipynb` to format the results.


`full_code/Mix-n-Match-Calibration` contains the IRM baseline (from [https://github.com/zhang64-llnl/Mix-n-Match-Calibration](https://github.com/zhang64-llnl/Mix-n-Match-Calibration)).

`full_code/PLMCalibration-main/` contains code for pre-trained language models (from [https://github.com/lifan-yuan/PLMCalibration](https://github.com/lifan-yuan/PLMCalibration)). We modified `prompt_ood.py` to export model outputs and data labels, which can then be used in `full_code/benchmark_calibration.py`.




