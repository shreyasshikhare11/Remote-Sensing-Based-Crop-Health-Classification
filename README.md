# Remote Sensing-Based Crop Health Classification NDVI and Fully Connected Neural Netwroks
https://arxiv.org/pdf/2504.10522

A precise agriculture pipeline that uses high-resolution remote sensing satellite imagery to classify crop health conditions. By integrating standard vegetation indices (NDVI, GNDVI, EVI, MSAVI) into a learnable Hybrid Vegetation Index (HVI) and tracking multi-temporal disease progression, this system utilizes a Fully Connected Neural Network (FCNN) to achieve 97.80% validation accuracy in detecting early-stage crop diseases like rust.

Multi-Feature Input Processing: Ingests raw multispectral/hyperspectral bands, automatically handling atmospheric noise and isolating vegetation from background noise (soil, rocks).
Learnable Hybrid Vegetation Index (HVI): Dynamically optimizes the fusion weights of multiple indices (NDVI, GNDVI, EVI, MSAVI) directly inside the neural network's backpropagation loop.
Multi-Temporal Analysis: Tracks time-series data to trace physiological degradation over a timeline, reducing false negatives in early disease stages.
Production-Ready FCNN: Includes a robust classification architecture utilizing localized features, non-linear activations (ReLU), and structural regularization (Dropout $p=0.5$) to eliminate spatial overfitting.
Direct Business Impact: Generates targeted field condition maps to minimize resource waste (fertilizers/pesticides) and eliminate labor-intensive manual inspections.

PROPOSED MODEL
[Data Acquisition Block] -> fetch_satellite_data()

Programmatically ingests multi-spectral Sentinel-2 Surface Reflectance (SR) data.
[Hybrid Multi-Index Integration Block] -> calculate_indices()

Integrates NDVI (Vigor), GNDVI (Green/Chlorophyll), EVI (Biomass), MSAVI (Soil-Adjusted), and NDRE (Red-Edge) to capture early chlorophyll degradation.
[Data Pre-Processing Block] -> mask_s2_clouds(), MNDWI Filter, PCHIP Interpolation

Implements pixel-level cloud/cirrus masking (QA60), MNDWI-based water extraction, and robust PCHIP temporal gap-filling to preserve trajectory continuity.
[Segmentation & Feature Extraction Block] -> create_classification_sequences() & LearnableHVIFusion()

Segments time-series into sliding temporal windows and extracts the dynamically-fused HVI.
[Fully Connected Neural Network Block] -> BiGRUAttentionClassifier()

Employs a state-of-the-art Bidirectional GRU + Attention sequence encoder feeding directly into the regularized Fully Connected Classifier (FCNN) network.
[Classification Result Block] -> Spectral Differentiation Output & Spatial Mapping

Segments and classifies pixels into: Healthy, Rust-Infected, and Severely Stressed.
