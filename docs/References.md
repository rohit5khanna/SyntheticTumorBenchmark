# References

This is a living reference list for the longitudinal tumor forecasting / synthetic benchmark project.

Purpose:

- keep all papers mentioned so far in one place,
- make later paper writing easier,
- track which papers are background, direct comparators, or benchmark-design references,
- and leave room for short notes as we read more carefully.

## How To Use This File

For future additions, try to include:

- full title
- venue / year if known
- link
- why it matters for this project
- optional short note after reading

---

## A. Longitudinal Tumor Forecasting / Imaging

### 1. Treatment-Aware Diffusion Probabilistic Model for Longitudinal MRI Generation and Diffuse Glioma Growth Prediction

- Venue / year: IEEE TMI, 2025
- Link: https://doi.org/10.1109/TMI.2025.3533038
- Why it matters:
  - direct comparator in longitudinal tumor forecasting
  - treatment-aware future mask and MRI prediction
  - uncertainty-aware generative forecasting

### 2. ImageFlowNet

- Link: https://arxiv.org/abs/2406.14794
- Why it matters:
  - longitudinal disease progression / image forecasting
  - useful as a stronger deep comparator family

### 3. Segmenting the Future

- Link: https://arxiv.org/abs/1904.10666
- Why it matters:
  - future segmentation forecasting
  - useful reference for direct future-mask prediction framing

### 4. Segis-Net

- Link: https://arxiv.org/abs/2012.14230
- Why it matters:
  - longitudinal segmentation / registration consistency
  - useful for temporal-consistency and prior-guided forecasting ideas

### 5. Vestibular schwannoma growth prediction from longitudinal MRI by time conditioned neural fields

- Link: https://arxiv.org/abs/2404.02614
- Why it matters:
  - neural-field based longitudinal growth prediction
  - relevant to shape-field / probability-field thinking

### 6. DELTA-MRI: Direct deformation Estimation from LongiTudinally Acquired k-space data

- Link: https://arxiv.org/abs/2301.09455
- Why it matters:
  - deformation / growth-field style longitudinal modeling
  - relevant to spatial change-field interpretations

### 7. Learning and forecasting tumor dynamics from longitudinal data

- Venue / year: PLOS Computational Biology, 2022
- Link: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009822
- Why it matters:
  - domain grounding for tumor dynamics forecasting
  - useful for framing tumor forecasting as an active research area

---

## B. Synthetic Data / Benchmarking / Evaluation

### 8. medGAN

- Venue / year: PMLR, 2017
- Link: https://proceedings.mlr.press/v68/choi17a.html
- Why it matters:
  - synthetic healthcare data generation baseline

### 9. Modeling Tabular Data using Conditional GAN (CTGAN)

- Venue / year: NeurIPS, 2019
- Link: https://papers.nips.cc/paper/8953-modeling-tabular-data-using-conditional-gan
- Why it matters:
  - synthetic data generation baseline

### 10. Time-series Generative Adversarial Networks (TimeGAN)

- Venue / year: NeurIPS, 2019
- Link: https://proceedings.neurips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html
- Why it matters:
  - sequential synthetic data generation baseline

### 11. TabDDPM

- Venue / year: PMLR / ICML, 2023
- Link: https://proceedings.mlr.press/v202/kotelnikov23a.html
- Why it matters:
  - diffusion-based synthetic data generation reference

### 12. PATE-GAN

- Link: https://openreview.net/forum?id=S1zk9iRqF7
- Why it matters:
  - privacy-aware synthetic data generation

### 13. Generation and evaluation of synthetic patient data

- Venue / year: BMC Medical Research Methodology, 2020
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC7204018/
- Why it matters:
  - fidelity / utility framing for synthetic healthcare data

### 14. Harnessing the power of synthetic data in healthcare

- Venue / year: npj Digital Medicine, 2023
- Link: https://www.nature.com/articles/s41746-023-00927-3
- Why it matters:
  - broad synthetic-data-in-healthcare perspective

### 15. Scoping review of privacy and utility metrics in medical synthetic data

- Venue / year: npj Digital Medicine, 2024
- Link: https://www.nature.com/articles/s41746-024-01359-3
- Why it matters:
  - evaluation framework for realism / privacy / utility

### 16. Membership inference attacks against synthetic health data

- Venue: Journal of Biomedical Informatics
- Link: https://www.sciencedirect.com/science/article/pii/S1532046421003063
- Why it matters:
  - privacy evaluation reference

### 17. A Kernel Two-Sample Test

- Venue / year: JMLR, 2012
- Link: https://jmlr.csail.mit.edu/papers/v13/gretton12a.html
- Why it matters:
  - MMD reference for distribution comparison

---

## C. Tumor Growth / Mechanistic Modeling / Parameter Inference

### 18. Image-driven parameter estimation for reaction-diffusion glioma model with mass effect

- Link: https://pubmed.ncbi.nlm.nih.gov/18026731/
- Why it matters:
  - mechanistic growth calibration reference

### 19. Inverse problem formulation for reaction-diffusion glioma parameter estimation

- Link: https://pubmed.ncbi.nlm.nih.gov/25963601/
- Why it matters:
  - parameter inference / calibration reference

### 20. Modeling glioma growth with mass effect by longitudinal MRI

- Link: https://pubmed.ncbi.nlm.nih.gov/34061731/
- Why it matters:
  - longitudinal mechanistic modeling reference

### 21. Limits of predictability in MRI-based mathematical tumor growth models

- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC12564088/
- Why it matters:
  - useful for honest limitations framing

### 22. Classical mathematical models for tumor growth

- Venue / year: PLOS Computational Biology, 2014
- Link: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003800
- Why it matters:
  - growth-model assumptions and tumor dynamics background

---

## D. Longitudinal / Survival / Interpretable Modeling

### 23. DeepHit

- Venue / year: AAAI, 2018
- Link: https://ojs.aaai.org/index.php/AAAI/article/view/11842
- Why it matters:
  - longitudinal / survival prediction baseline family

### 24. Dynamic-DeepHit

- Venue / year: IEEE TBME, 2020
- Link: https://pubmed.ncbi.nlm.nih.gov/30951460/
- Why it matters:
  - dynamic risk prediction from longitudinal trajectories

### 25. Joint modeling overview

- Venue: Annual Review of Statistics and Its Application
- Link: https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-030718-105048
- Why it matters:
  - longitudinal plus event modeling reference

### 26. RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism

- Venue / year: NeurIPS, 2016
- Link: https://papers.neurips.cc/paper/6321-retain-an-interpretable-predictive-model-for-healthcare-using-reverse-time-attention-mechanism
- Why it matters:
  - interpretable temporal healthcare modeling

### 27. BEHRT

- Venue / year: Scientific Reports, 2020
- Link: https://www.nature.com/articles/s41598-020-62922-y
- Why it matters:
  - transformer-style healthcare sequence modeling

### 28. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead

- Author: Cynthia Rudin
- Venue / year: Nature Machine Intelligence, 2019
- Link: https://www.nature.com/articles/s42256-019-0048-x
- Why it matters:
  - interpretability philosophy reference

### 29. A Unified Approach to Interpreting Model Predictions (SHAP)

- Venue / year: NeurIPS, 2017
- Link: https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- Why it matters:
  - feature attribution baseline

### 30. Calibration: the Achilles heel of predictive analytics

- Venue / year: BMC Medicine, 2019
- Link: https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-019-1466-7
- Why it matters:
  - calibration-first evaluation reminder

### 31. TRIPOD Statement

- Venue / year: Annals of Internal Medicine, 2015
- Link: https://www.acpjournals.org/doi/10.7326/M14-0697
- Why it matters:
  - reporting checklist for predictive modeling studies

---

## E. Project-Specific Comparator / Benchmark Notes

### 32. TaDiff-Net paper / repo

- Paper link: https://doi.org/10.1109/TMI.2025.3533038
- Repo: local project reference in `TaDiff-Net`
- Why it matters:
  - direct real-data longitudinal comparator family

### 33. SAILOR processed longitudinal glioma dataset

- Not a paper entry by itself here; dataset context used for:
  - realism checks
  - eventual external validation

---

## F. To Be Added Later

Use this section for papers we discuss later but have not yet fully categorized.

- `TBD`
