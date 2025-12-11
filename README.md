# Estimating uncertainty of needle localization during Transrectal Ultrasound-guided High-Dose-Rate Prostate Brachytherapy
<img width="612" height="218" alt="ucert-mcdo" src="https://github.com/user-attachments/assets/a9e02576-dd86-4787-9acc-2e12af7ddc18" />

### Abstract
Purpose: Accurate identification of needle spots in transrectal ultrasound (TRUS) is key for safe and efficient high–dose-rate (HDR) prostate brachytherapy. We present a probability-based segmentation framework that quantifies pixel-level uncertainty via Monte Carlo Dropout (MCDO) and enables fully configurable loss functions driven by a single configuration file.

Methods: The pipeline uses axial TRUS images with mask supervision (PNG). We integrate MCDO at test time to generate uncertainty maps from multiple stochastic passes and aggregate predictions into probability maps for threshold-based detection. Training uses MONAI 1.3.0 with a config-driven loss selector (e.g., DiceCE, Focal, Tversky) and hyperparameter search via Random, Grid, and Bayesian optimization (Optuna). We export per-slice uncertainty summaries to CSV for downstream analysis.

Results: On held-out test slices, uncertainty concentrated near shadowed regions and needle–tissue boundaries, aligning with common failure modes. MCDO-based aggregation reduced spurious detections in noisy areas compared to deterministic inference. Stratifying by uncertainty improved error analysis and guided threshold selection.

Conclusion: MCDO provides actionable uncertainty that clarifies when the model is confident. A config-driven design makes loss selection and inference modes easy to tune, and Optuna expands the search space to thresholds and loss hyperparameters. The approach preserves existing training behavior while adding uncertainty-aware testing and analysis.

Keywords: HDR prostate brachytherapy, TRUS, segmentation, Monte Carlo Dropout, uncertainty, MONAI, Optuna, Tversky loss, Focal loss

### Introduction
TRUS-guided HDR brachytherapy relies on precise needle placement to deliver radiation safely to the prostate while sparing surrounding tissue. Reliable localization of needle spots on axial TRUS images helps operators confirm placement and supports intraoperative planning. Yet, ultrasound speckle, acoustic shadowing, and anatomical variability make segmentation difficult. Classical U-Net–style convolutional networks have advanced medical image segmentation, but estimating when the model is unsure remains important for clinical use and quality assurance [3]. In this work, we implement probability-based inference using Monte Carlo Dropout (MCDO) to evaluate predictive uncertainty [1], and we keep training behavior unchanged. We also make the cost function fully configurable (e.g., DiceCE, Focal, Tversky), since different losses can shift the balance between overlap, class imbalance, and boundary sensitivity [7], [8]. The entire pipeline is built on MONAI 1.3.0 [4] and uses Optuna for hyperparameter search [5].

Uncertainty estimation has emerged as a practical tool for decision support in medical imaging. Approximate Bayesian inference with dropout provides a simple, scalable route to capture epistemic uncertainty in deep networks [1], and combining it with task-aware losses and calibration yields more trustworthy predictions [2], [11]. In prostate brachytherapy, procedural guidance standards underscore the need for reliable imaging feedback during needle placement [9]. We focus on axial TRUS images with expert-annotated PNG masks of needle spots and ask a simple question: can we keep the training recipe stable, add MCDO at test time, and gain useful uncertainty maps and better threshold selection without complicating clinical workflows?

Our contributions are practical and aimed at researchers and clinical engineers. We expose a config toggle to switch between standard inference and MCDO, export per-slice uncertainty summaries for analysis, and extend the search space to include loss types and decision thresholds. The result is a reproducible, uncertainty-aware segmentation workflow that fits into a typical MONAI project while supporting exploration across losses and inference modes.

### Uniqueness and Novelty
- Config-driven uncertainty: A single configuration controls inference mode (standard vs. MCDO), number of stochastic passes, probability thresholding, and loss selection—no code changes needed.
- Uncertainty to CSV: We export per-slice uncertainty summaries (e.g., variance, entropy, foreground probability statistics) to CSV, enabling quick stratification of hard slices and distance-to-error analyses.
- Searchable decision rules: We extend Optuna’s search space to include loss families and probability thresholds, letting Bayesian optimization jointly tune learning and decision parameters [5].
- Training preserved, testing enriched: The training pipeline and data loaders remain intact; MCDO is applied only at test time, which simplifies adoption and avoids perturbing established training behavior.

### Methods
We train a 2D segmentation model on axial TRUS images from HDR prostate brachytherapy procedures. Ground-truth needle spot masks are PNGs aligned to input images. The codebase uses MONAI 1.3.0 transforms for standard preprocessing and augmentation. We keep the original model architecture and training schedule intact and introduce the following additions:

- Monte Carlo Dropout for inference: We enable dropout layers during testing and run N stochastic forward passes per slice. The mean of the probability maps serves as the final estimate, and pixelwise dispersion (variance and predictive entropy) quantifies uncertainty [1], [2]. This strategy is simple to deploy, runs on the existing network, and avoids architectural changes.

- Probability-based detection: We use the aggregated probability map and apply a configurable threshold to generate the final mask. Because thresholds interact with calibration, we include the threshold as a searchable hyperparameter. This turns “how confident is confident enough?” into an empirical question optimized on validation data.

- Configurable cost functions: We expose a loss selector with configurable hyperparameters. Options include:
  - Dice + Cross-Entropy (DiceCE) for overlap and class balance,
  - Focal loss for hard-example emphasis via γ [8],
  - Tversky loss to tilt penalties between false positives and false negatives via α, β [7],
  - BCE or composite losses as needed.  
  This flexibility matters for small, sparse targets like needle spots, where class imbalance and boundary sharpness strongly influence outcomes.

- Hyperparameter search: We retain Random and Grid search and use Optuna for Bayesian optimization [5]. The search space now includes loss type, loss parameters (e.g., γ, α, β), dropout rate, and probability threshold. We keep priors broad but clinically plausible to avoid overfitting small validation sets.

- Uncertainty summaries: For each test slice, we compute summary statistics of uncertainty (e.g., mean/median variance and entropy within predicted foreground and background) and write these to CSV. These summaries help analysts triage difficult slices, visualize failure modes, and link uncertainty to performance.

### Evaluation
We perform patient-wise splits to avoid leakage. The test set includes axial TRUS slices with varying gland size and shadowing. We evaluate segmentation performance and the utility of uncertainty estimates.

- Segmentation metrics: We report Dice, Jaccard/IoU, precision, recall, and F1 to capture overlap and detection quality [6]. Where relevant, we use 95th percentile Hausdorff distance (95HD) to reflect boundary errors.

- Decision threshold analysis: Because thresholding directly affects foreground extent in sparse targets, we treat the probability threshold as a hyperparameter and evaluate operating points on the validation set. We found it helpful to visualize precision–recall trade-offs and select a threshold that aligns with clinical tolerance for missed vs. spurious spots.

- Uncertainty quality: We examine the relationship between uncertainty and error using risk–coverage curves and error–uncertainty correlation [2]. We also review calibration with Expected Calibration Error (ECE) and Brier score [11], focusing on whether MCDO aggregation yields better-aligned probabilities for thresholding.

- Slice-level triage: Using the exported CSV, we summarize uncertainty by slice and stratify test performance by uncertainty quantiles. This is a simple diagnostic: if higher-uncertainty slices carry most errors, uncertainty becomes a practical flag for review.

All evaluations use the same deterministic preprocessing and test-time postprocessing across inference modes to ensure fair comparisons.

### Results
MCDO produced interpretable uncertainty maps that matched where the model struggled: near acoustic shadows, at the gland apex and base, and around faint echogenic lines that resemble needles. Compared with single-pass deterministic inference, aggregating multiple dropout-enabled passes smoothed the probability map and reduced salt-and-pepper artifacts. In practice, that made threshold selection easier and less sensitive to slice-to-slice noise.

Uncertainty stratification was informative. High-uncertainty slices contained a disproportionate share of false positives in speckled areas and ambiguous boundaries. Foreground entropy within predicted spots was a useful indicator of fragile detections. When we tuned the probability threshold with Optuna—jointly with the loss family and its hyperparameters—the final masks were more stable across patients, especially when the chosen loss was sensitive to class imbalance (e.g., Focal or Tversky).

From a workflow perspective, exporting per-slice CSV summaries made error analysis faster. Reviewing a handful of high-uncertainty slices pointed us toward consistent imaging patterns, like shadowing from the pubic arch or needles at oblique angles. Although we did not change the training recipe, simply switching to MCDO at test time yielded clearer probability maps and fewer spurious detections under the same postprocessing. This suggests that probability aggregation and uncertainty-informed thresholding can add value without architectural changes.

### Conclusion
We introduced a probability-aware segmentation framework for needle spot detection in axial TRUS that preserves existing training behavior while adding MCDO-based uncertainty at test time. The approach is entirely config-driven: users can toggle inference modes, export per-slice uncertainty to CSV, and swap among loss functions with tunable hyperparameters. Extending Bayesian optimization to decision thresholds and loss families helped align model behavior with clinical priorities.

Future work includes: (1) uncertainty calibration and temperature scaling to further stabilize thresholds, (2) test-time augmentation ensembles or deep ensembles to compare with MCDO, (3) joint modeling of gland and needle context to reduce ambiguity near boundaries, (4) extension to 3D or temporal stacks when cine TRUS is available, and (5) uncertainty-guided active learning to focus annotation on the most informative slices.

### References
[1] Y. Gal and Z. Ghahramani, “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,” in Proc. ICML, 2016, pp. 1050–1059.

[2] A. Kendall and Y. Gal, “What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?,” in Proc. NeurIPS, 2017, pp. 5574–5584.

[3] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” in Proc. MICCAI, 2015, pp. 234–241.

[4] MONAI Consortium, F. Cardoso et al., “MONAI: An Open-Source Framework for Deep Learning in Healthcare,” arXiv:2211.02701, 2022.

[5] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, “Optuna: A Next-Generation Hyperparameter Optimization Framework,” in Proc. KDD, 2019, pp. 2623–2631.

[6] A. A. Taha and A. Hanbury, “Metrics for Evaluating 3D Medical Image Segmentation: Analysis, Selection, and Tool,” BMC Medical Imaging, vol. 15, no. 29, 2015.

[7] S. S. M. Salehi, D. Erdogmus, and A. Gholipour, “Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks,” in MLMI, 2017, pp. 379–387.

[8] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal Loss for Dense Object Detection,” in Proc. ICCV, 2017, pp. 2980–2988.

[9] P. J. Hoskin et al., “GEC/ESTRO Recommendations on High-Dose-Rate Brachytherapy for Prostate Cancer,” Radiotherapy and Oncology, vol. 107, no. 3, pp. 325–332, 2013.

[10] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” in Proc. ICML, 2017, pp. 1321–1330.
