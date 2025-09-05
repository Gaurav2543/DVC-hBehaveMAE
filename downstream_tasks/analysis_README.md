# Comprehensive Behavioral Embedding Analysis Framework

## Overview

This framework provides a complete analysis pipeline for behavioral embeddings extracted from animal behavior data. It handles 24-hour temporal patterns (1440 time points = 1440 minutes) with multi-dimensional embeddings, providing both basic and enhanced analysis capabilities with optional bodyweight covariate integration.

## Architecture and Implementation

### Core Data Structure

The framework expects embeddings in the following format:
```python
embeddings_data = {
    'embeddings': np.ndarray,           # Shape: (total_frames, n_dimensions)
    'frame_number_map': dict,           # Maps sequence keys to (start, end) indices
    'aggregation_info': dict            # Metadata about temporal aggregation
}
```

**Key Assumptions:**
- Each sequence represents exactly 24 hours (1440 frames)
- Each frame represents 1 minute of behavior
- Embeddings are pre-extracted using a model like hBehaveMAE
- Frame mapping uses keys in format: "CAGE_ID_YYYY-MM-DD"

### Module Structure

#### 1. Basic Embedding Analyzer (`EmbeddingAnalyzer` class)

**Purpose**: Comprehensive analysis of behavioral embeddings with standard machine learning approaches.

**Core Components:**

##### Data Loading and Alignment
```python
def load_data(self):
    # Loads embeddings from .npy file (expects dictionary format)
    # Handles both old numpy format and new structured format
    # Creates aligned dataset using frame mapping
```

**Default Parameters:**
- `max_workers = 8`: Number of parallel threads for processing
- `test_size = 0.25`: Train/test split ratio for classification
- `n_samples = 1000`: Number of samples for activity heatmaps

##### Activity Heatmap Generation
```python
def create_activity_heatmaps(self, n_samples=1000):
    # Creates heatmaps for first 16 dimensions (configurable)
    # Each heatmap shows 24-hour patterns across multiple samples
    # Time axis: 1440 minutes converted to 24-hour format
```

**Implementation Details:**
- **Dimension Limit**: First 16 dimensions only (performance optimization)
- **Time Binning**: 60-minute bins for hour labels (positions at 0, 60, 120, ...)
- **Color Scheme**: 'viridis' colormap for better visibility
- **Output**: PNG (300 DPI) and PDF formats
- **Memory Management**: Processes samples in batches to avoid memory overflow

**What's Included:**
- Individual dimension heatmaps showing temporal patterns
- Average activity patterns across all samples
- Standard deviation bands for variability assessment
- Time-of-day labels in HH:MM format

**What's Excluded:**
- Statistical significance testing of temporal patterns
- Fourier analysis for frequency domain patterns
- Individual sample temporal anomaly detection

##### Age Regression Analysis
```python
def age_regression_analysis(self, include_bodyweight=True):
    # Performs comprehensive age prediction analysis
    # Uses Leave-One-Group-Out cross-validation by strain
```

**Default Parameters:**
- `alpha = 1.0`: Ridge regression regularization parameter
- `test_size = 0.2`: For simple train/test splits where LOGO not applicable
- Cross-validation: Leave-One-Group-Out by strain (no default k-fold)

**Models Used:**
1. **Ridge Regression** (primary): L2 regularization, handles multicollinearity
2. **Linear Regression**: Baseline comparison, no regularization
3. **Lasso Regression**: L1 regularization for feature selection

**Validation Strategy:**
- **Leave-One-Group-Out (LOGO)**: Each strain held out once as test set
- **Rationale**: Prevents data leakage between genetically related animals
- **Alternative**: Simple train/test split for insufficient strain diversity

**Metrics Computed:**
- **RMSE (Root Mean Square Error)**: Primary regression metric
- **R² (Coefficient of Determination)**: Explains variance
- **MAE (Mean Absolute Error)**: Robust to outliers
- **Baseline MAE**: Using mean age prediction as baseline

**Implementation Details:**
- **Feature Scaling**: StandardScaler applied to all features
- **Missing Data**: Samples with NaN values removed before analysis
- **Bodyweight Integration**: Appended as additional feature when available
- **Parallel Processing**: Each fold processed independently

**What's Included:**
- Full model regression (all dimensions together)
- Dimension-wise regression (each dimension separately)
- Feature selection approaches (SelectKBest, RFE, Lasso, RandomForest)
- Per-strain performance analysis
- Sliding window analysis across age ranges

**What's Excluded:**
- Non-linear regression models (SVM, neural networks)
- Time series specific methods (ARIMA, state space models)
- Hierarchical modeling accounting for cage effects
- Bootstrap confidence intervals
- Permutation testing for significance

##### Strain Classification Analysis
```python
def strain_classification_analysis(self):
    # Multi-class classification of strain identity
    # Uses standard train/test split due to strain imbalance
```

**Default Parameters:**
- `test_size = 0.2`: Train/test split ratio
- `random_state = 42`: Reproducible results
- `max_iter = 1000`: Logistic regression convergence limit
- `n_estimators = 100`: Random Forest tree count

**Models Used:**
1. **Logistic Regression** (primary): Linear decision boundaries
2. **Random Forest Classifier**: Non-linear, ensemble method

**Validation Strategy:**
- **Stratified Train/Test Split**: Maintains strain proportions
- **Rationale**: LOGO not suitable due to single test strain per fold

**Metrics Computed:**
- **Accuracy**: Overall classification correctness
- **F1 Score (Weighted)**: Accounts for class imbalance
- **Per-Class Accuracy**: Individual strain performance
- **Classification Report**: Precision, recall, F1 per strain

**What's Included:**
- Full model classification (all dimensions)
- Dimension-wise classification (individual predictive power)
- Confusion matrix visualization
- Per-strain performance analysis

**What's Excluded:**
- Multi-label classification (single strain per sample assumed)
- Hierarchical classification (strain families/genetics)
- Cost-sensitive learning for imbalanced classes
- Cross-validation (simple split used for speed)

##### PCA Analysis
```python
def perform_pca_analysis(self):
    # Principal Component Analysis for dimensionality reduction
    # Analyzes variance explained and component interpretability
```

**Default Parameters:**
- `n_components = None`: All components computed initially
- `n_components_plot = 20`: First 20 components shown in plots
- `scaler = StandardScaler()`: Z-score normalization

**Analysis Components:**
1. **Scree Plot**: Variance explained per component
2. **Cumulative Variance**: Running total of explained variance
3. **Component Loadings**: Original dimension contributions
4. **PC Scatter Plots**: 2D/3D visualization colored by age/strain

**What's Included:**
- Explained variance ratios for all components
- Component loadings matrix (first 10 PCs × first 20 dimensions)
- PC correlations with age and strain
- Interactive scatter plots (PC1 vs PC2, PC1 vs PC3)
- Statistical significance of PC-age correlations

**What's Excluded:**
- Sparse PCA for interpretable components
- Factor analysis for latent variable modeling
- Independent Component Analysis (ICA)
- Non-linear dimensionality reduction (t-SNE, UMAP)

##### Clustering Analysis
```python
def perform_clustering_analysis(self):
    # Multiple clustering algorithms with evaluation metrics
    # Uses PCA-reduced features for computational efficiency
```

**Default Parameters:**
- `n_components_pca = 50`: PCA reduction before clustering
- `eps = 0.5`: DBSCAN density parameter
- `min_samples = 5`: DBSCAN minimum cluster size

**Algorithms Used:**
1. **K-Means**: k = 5, 10, 15 clusters
2. **Hierarchical Clustering**: Agglomerative, k = 5, 10
3. **DBSCAN**: Density-based, automatic cluster number

**Evaluation Metrics:**
- **Silhouette Score**: Cluster quality measure
- **Cluster Size Distribution**: Number of samples per cluster
- **Age/Strain Distribution**: Composition analysis per cluster

**What's Included:**
- Interactive HTML plots with hover information (age, strain, cage ID)
- Cluster characteristics analysis (age ranges, strain composition)
- Silhouette score comparison across algorithms
- Cluster centroid analysis in embedding space

**What's Excluded:**
- Gaussian Mixture Models (probabilistic clustering)
- Spectral clustering for non-convex clusters
- Cluster validation using external criteria
- Hierarchical cluster dendrograms
- Optimal cluster number selection (elbow method, etc.)

##### Sliding Window Analysis
```python
def perform_sliding_window_regression(self, X, y, groups, suffix):
    # Age regression within overlapping age windows
    # Identifies age-dependent embedding changes
```

**Default Parameters:**
- `window_sizes = [30, 60, 90]`: Window sizes in days
- `window_step = 15`: Step size in days
- `min_samples = 20`: Minimum samples per window
- `test_size = 0.3`: Train/test split within each window

**Implementation Details:**
- **Ridge Regression**: alpha=1.0 used within each window
- **Feature Scaling**: StandardScaler applied per window
- **Age Binning**: Based on 'avg_age_days_chunk_start' metadata

**Outputs:**
- **RMSE vs Age Window**: Performance across age ranges
- **R² vs Age Window**: Variance explained across ages
- **Sample Count**: Number of samples per window
- **Window Center**: Midpoint of each age window

**What's Included:**
- Multiple window sizes for temporal resolution comparison
- Overlapping windows for smooth temporal analysis
- Performance visualization across age ranges
- Data export for external plotting/analysis

**What's Excluded:**
- Varying step sizes for different window sizes
- Statistical significance testing between windows
- Temporal autocorrelation analysis
- Changepoint detection in age relationships

##### Anomaly Detection
```python
def detect_anomalous_samples(self):
    # Identifies samples that cluster poorly with their strain
    # Uses Mahalanobis-like distance from strain centroids
```

**Default Parameters:**
- `threshold_multiplier = 2.5`: Standard deviations for outlier detection
- `min_strain_samples = 5`: Minimum samples required per strain

**Algorithm:**
1. Compute strain-wise centroids in embedding space
2. Calculate Euclidean distances from each sample to its strain centroid
3. Identify samples beyond threshold × standard deviation

**What's Included:**
- Anomalous sample identification with metadata
- Distance from strain centroid for each outlier
- Export to both JSON and human-readable text formats
- Sample information: cage ID, age, date, strain

**What's Excluded:**
- Isolation Forest or other ensemble anomaly detection
- Local Outlier Factor (LOF) for local density anomalies
- Temporal anomaly detection (unusual daily patterns)
- Multi-modal anomaly detection (age + embedding space)

#### 2. Enhanced Analyzer with Bodyweight Integration

##### Bodyweight Data Integration
```python
class BodyweightHandler:
    # Handles bodyweight data loading and temporal matching
```

**Data Sources:**
1. **BW.csv**: Regular bodyweight measurements
2. **TissueCollection_PP.csv**: Pre-sacrifice weights
3. **Summary_metadata.csv**: Animal-cage mapping with timepoints

**Matching Strategies:**

1. **Closest (`closest`)** - Default:
   ```python
   # Finds measurement closest in time to target date
   # Simple temporal distance minimization
   # Fast, works well for frequent measurements
   ```

2. **Linear Interpolation (`interpolate`)**:
   ```python
   # Interpolates between surrounding measurements
   # Requires at least 2 measurements per animal
   # More accurate for sparse measurements
   ```

3. **Most Recent (`most_recent`)**:
   ```python
   # Uses most recent measurement before target date
   # Conservative approach, avoids future data leakage
   # Good for growth trend analysis
   ```

4. **GAM Spline (`gam_spline`)**:
   ```python
   # Generalized Additive Model with smooth splines
   # Requires at least 3 measurements per animal
   # Handles non-linear growth patterns
   # Most sophisticated but computationally expensive
   ```

**Default Parameters:**
- `strategy = "gam_spline"`: Default matching strategy
- `cage_aggregation = "mean"`: How to combine multiple animals per cage
- `n_splines = min(10, len(data)-1, max(3, len(data)//3))`: GAM spline count

**Implementation Details:**
- **Missing Data**: NaN returned for insufficient data
- **Outlier Handling**: GAM predictions validated against reasonable ranges
- **Parallel Processing**: ThreadPoolExecutor for faster matching
- **Memory Efficiency**: Processes animals individually to avoid large matrices

**What's Included:**
- Multiple temporal matching strategies
- Cage-level aggregation across multiple animals
- Robust error handling for missing/sparse data
- Progress tracking for large datasets

**What's Excluded:**
- Hierarchical modeling of growth curves
- Seasonal/circadian bodyweight adjustments
- Sex-specific growth modeling
- Uncertainty quantification in interpolation

##### Comprehensive Temporal Analysis
```python
def perform_comprehensive_temporal_analysis(self):
    # Analyzes 24-hour patterns across multiple dimensions
```

**Components:**

1. **Circadian Rhythm Analysis**:
   - **Temporal Aggregation**: 24 hourly averages per sample
   - **Statistical Summary**: Mean ± standard deviation across samples
   - **Visualization**: Individual dimension plots with confidence bands
   - **Data Storage**: JSON export for external analysis

2. **Temporal Dimension Importance**:
   - **Temporal Variance**: Within-sample hour-to-hour variation
   - **Circadian Amplitude**: Max - Min of average 24-hour pattern
   - **Ranking**: Dimensions sorted by temporal variation
   - **Visualization**: Bar plots with color-coded importance

3. **Age-Temporal Relationships**:
   - **Age Binning**: Tertiles (Young, Middle, Old)
   - **Group Comparison**: Temporal patterns by age group
   - **Statistical Testing**: Not implemented (excluded feature)

4. **Strain-Temporal Patterns**:
   - **Strain Selection**: Top 6 strains by sample count
   - **Pattern Comparison**: Overlay plots across strains
   - **Minimum Sample Size**: 5 samples per strain required

**Default Parameters:**
- `n_dims_circadian = 16`: Dimensions analyzed for circadian patterns
- `n_dims_temporal_importance = all`: All dimensions ranked
- `age_percentiles = [0, 33, 66, 100]`: Age group boundaries
- `min_strain_samples = 5`: Minimum for strain pattern analysis

**What's Included:**
- Hour-by-hour pattern analysis across dimensions
- Statistical summaries (mean, std) of temporal patterns
- Age-dependent temporal pattern changes
- Strain-specific temporal signatures
- Comprehensive visualization suite

**What's Excluded:**
- Fourier analysis for frequency domain patterns
- Phase shift detection between strains/ages
- Statistical significance testing of temporal differences
- Seasonal or long-term temporal trend analysis
- Individual sample temporal anomaly detection

##### Enhanced Correlation Analysis
```python
def create_comprehensive_correlation_analysis(self):
    # Multi-variable correlation analysis with visualization
```

**Correlation Matrix Components:**
1. **Age**: Primary target variable
2. **Bodyweight**: Optional covariate (if available)
3. **Strain**: Encoded as numeric for correlation
4. **Embedding Dimensions**: All dimensional features

**Visualization Components:**
1. **Full Correlation Matrix**: Heatmap of all variables
2. **Age Correlations**: Top 20 dimensions by |correlation| with age
3. **Bodyweight Correlations**: Top 20 dimensions correlated with bodyweight
4. **Inter-Dimension Correlations**: Embedding dimension relationships

**Default Parameters:**
- `top_correlations = 20`: Number of top correlations displayed
- `correlation_threshold`: Not implemented (shows all)
- `significance_testing`: Not implemented (excluded feature)

**Color Coding:**
- **Age Correlations**: Red (positive) to Blue (negative)
- **Bodyweight Correlations**: Green (positive) to Purple (negative)
- **Statistical Significance**: Not color-coded (excluded feature)

**What's Included:**
- Comprehensive correlation matrix computation
- Top predictive dimensions identification
- Multi-variable relationship visualization
- JSON export of correlation coefficients

**What's Excluded:**
- Partial correlations controlling for confounders
- Statistical significance testing (p-values)
- Correction for multiple comparisons
- Non-linear correlation measures (Spearman, Kendall)
- Causal inference methods

##### Enhanced Feature Importance Analysis
```python
def perform_enhanced_feature_importance_analysis(self):
    # Multi-method feature importance with consensus ranking
```

**Methods Implemented:**

1. **Univariate Correlations**:
   - **Pearson Correlation**: Linear relationship with age
   - **Statistical Testing**: P-values computed but not corrected
   - **Significance Threshold**: p < 0.05 for visualization

2. **Ridge Regression Coefficients**:
   - **Regularization**: alpha = 1.0
   - **Coefficient Magnitude**: |coefficient| used for ranking
   - **Feature Scaling**: StandardScaler applied before fitting

3. **Random Forest Importance**:
   - **Trees**: n_estimators = 100
   - **Importance Measure**: Mean decrease in impurity
   - **Random State**: 42 for reproducibility

4. **Consensus Ranking**:
   - **Method**: Average rank across all methods
   - **Ties**: Handled by average ranking
   - **Top Features**: Lowest average rank = most important

**Default Parameters:**
- `ridge_alpha = 1.0`: L2 regularization strength
- `rf_n_estimators = 100`: Number of trees
- `rf_random_state = 42`: Reproducibility seed
- `top_features_display = 20`: Features shown in plots

**What's Included:**
- Multiple complementary importance measures
- Consensus ranking across methods
- Statistical significance indicators
- Comprehensive visualization comparing methods
- JSON export of all importance scores

**What's Excluded:**
- Permutation importance (computationally expensive)
- SHAP (SHapley Additive exPlanations) values
- Recursive feature elimination with cross-validation
- Stability assessment across bootstrap samples
- Feature interaction importance

##### Interactive Visualization Dashboard
```python
def create_interactive_exploration_dashboard(self):
    # Interactive HTML plots using Plotly
```

**Components:**

1. **3D PCA Plot**:
   - **Dimensions**: PC1, PC2, PC3
   - **Color Coding**: By strain (if available) or age
   - **Hover Information**: Age, strain, cage ID
   - **Interactivity**: Rotation, zoom, pan, point selection

2. **2D PCA with Age Groups**:
   - **Dimensions**: PC1 vs PC2
   - **Age Binning**: Quartiles for group visualization
   - **Color Coding**: By age group
   - **Hover Information**: Individual age values

**Default Parameters:**
- `pca_components = min(10, n_features)`: PCA components computed
- `plot_dimensions = 3`: 3D visualization primary
- `age_groups = 4`: Quartile-based age binning
- `marker_size = 5-6`: Point size in plots
- `opacity = 0.7`: Point transparency

**File Outputs:**
- `interactive_3d_pca.html`: Main 3D exploration plot
- `interactive_2d_pca_age.html`: Age-focused 2D plot

**What's Included:**
- Fully interactive 3D scatter plots
- Hover tooltips with metadata
- Strain/age color coding
- PCA explained variance labels
- Responsive web-based visualization

**What's Excluded:**
- Interactive dimension selection
- Real-time clustering updates
- Animation through time/age
- Linked brushing between multiple plots
- Custom color scheme selection

### Command Line Interface

#### Usage Patterns

**Basic Analysis**:
```bash
python run_analysis.py embeddings.npy metadata.csv
```

**With Bodyweight**:
```bash
python run_analysis.py embeddings.npy metadata.csv \
  --bodyweight-file BW.csv \
  --tissue-file TissueCollection_PP.csv \
  --summary-file summary_metadata.csv
```

**Performance Options**:
```bash
python run_analysis.py embeddings.npy metadata.csv \
  --quick-mode \
  --max-workers 4 \
  --skip-interactive
```

#### Default Parameter Values

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--output-dir` | "embedding_analysis_results" | Base output directory |
| `--bodyweight-strategy` | "closest" | Temporal matching method |
| `--analysis-type` | "both" | Run basic + enhanced analysis |
| `--max-workers` | 8 | Parallel processing threads |
| `--n-heatmap-samples` | 100 | Samples for heatmap visualization |
| `--quick-mode` | False | Reduced processing for testing |
| `--skip-interactive` | False | Skip Plotly HTML generation |
| `--skip-temporal` | False | Skip temporal pattern analysis |

#### File Validation

**Input Validation**:
- Embeddings file existence and format
- Metadata CSV structure
- Bodyweight file availability (if specified)
- Required columns in metadata

**Error Handling**:
- Graceful degradation for missing optional files
- Informative error messages with suggested fixes
- Automatic fallback to basic analysis if enhanced fails

### Output Structure and File Formats

#### Directory Organization
```
embedding_analysis_results/
├── basic_analysis/
│   ├── heatmaps/
│   │   ├── activity_heatmap_dim_*.png/pdf
│   │   └── average_activity_patterns.png/pdf
│   ├── regression/
│   │   ├── full_model_results_*.json
│   │   ├── dimensionwise_results_*.json
│   │   └── strain_regression_results_*.png/pdf
│   ├── classification/
│   │   ├── full_model_strain_results.json
│   │   ├── dimensionwise_strain_results.json
│   │   └── *_class_accuracy.png/pdf
│   ├── feature_selection/
│   │   ├── feature_selection_results_*.json
│   │   └── feature_selection_*.png/pdf
│   ├── pca_analysis/
│   │   ├── pca_results.json
│   │   ├── pca_overview.png/pdf
│   │   └── component_loadings.png/pdf
│   ├── clustering/
│   │   ├── clustering_results.json
│   │   ├── cluster_characteristics.json
│   │   └── interactive_clustering_*.html
│   ├── sliding_window/
│   │   ├── sliding_window_results_*.json
│   │   └── sliding_window_*.png/pdf
│   └── anomaly_detection/
│       ├── anomalous_samples.json
│       └── anomalous_samples.txt
└── enhanced_analysis/
    ├── temporal_analysis/
    │   ├── circadian_data.json
    │   ├── temporal_importance.json
    │   ├── circadian_rhythms.png/pdf
    │   ├── temporal_dimension_importance.png/pdf
    │   ├── age_temporal_relationships.png/pdf
    │   └── strain_temporal_patterns.png/pdf
    ├── correlation_analysis/
    │   ├── correlation_results.json
    │   └── comprehensive_correlations.png/pdf
    ├── feature_importance/
    │   ├── feature_importance_results.json
    │   └── feature_importance_comparison.png/pdf
    └── interactive_plots/
        ├── interactive_3d_pca.html
        └── interactive_2d_pca_age.html
```

#### File Format Specifications

**JSON Files**:
- **Purpose**: Numerical results, parameters, metadata
- **Structure**: Hierarchical dictionaries with typed values
- **Encoding**: UTF-8, human-readable formatting
- **Precision**: Float values preserved to full precision

**PNG Files**:
- **Resolution**: 300 DPI for publication quality
- **Format**: 24-bit RGB
- **Compression**: Lossless PNG compression
- **Size**: Variable based on plot complexity

**PDF Files**:
- **Purpose**: Vector graphics for publications
- **Standard**: PDF 1.4 compatible
- **Fonts**: Embedded for portability
- **Resolution**: Vector (infinite scalability)

**HTML Files**:
- **Library**: Plotly.js embedded
- **Compatibility**: Modern browsers (Chrome, Firefox, Safari)
- **Interactivity**: Full zoom, pan, hover, selection
- **Size**: Self-contained (no external dependencies)

**TXT Files**:
- **Purpose**: Human-readable summaries
- **Encoding**: UTF-8
- **Format**: Structured text with clear delimiters
- **Content**: Anomalous samples, key findings

### Memory and Performance Characteristics

#### Memory Usage Patterns

**Peak Memory Usage**:
- **Embeddings Loading**: 1x embedding size (your case: ~9.1 GB)
- **Temporal Patterns**: 24x embedding size (24 hourly patterns stored)
- **PCA Analysis**: 2-3x embedding size (original + transformed data)
- **Total Estimate**: 25-30x embedding size during temporal analysis

**Memory Optimization Strategies**:
- **Batch Processing**: Heatmaps processed in batches of 100 samples
- **Data Type Optimization**: float16 used where precision allows
- **Garbage Collection**: Explicit cleanup after large operations
- **Progressive Analysis**: Components run sequentially, not simultaneously

#### Computational Complexity

**Time Complexity by Component**:
- **Data Loading**: O(n) where n = number of samples
- **Heatmap Generation**: O(k × m × d) where k=samples, m=timepoints, d=dimensions
- **Age Regression**: O(s × n × d²) where s=strains for LOGO CV
- **PCA Analysis**: O(n × d²) for full decomposition
- **Clustering**: O(n² × d) for hierarchical, O(k × n × d × i) for k-means
- **Temporal Analysis**: O(n × 24 × d) for hourly patterns

**Parallelization Strategy**:
- **ThreadPoolExecutor**: I/O bound operations (file processing)
- **Process-based**: Not used (memory overhead too high)
- **Vectorized Operations**: NumPy/scikit-learn optimized routines
- **Default Workers**: 8 threads (adjustable via --max-workers)

#### Performance Benchmarks

**Typical Runtime (approximate, depends on hardware)**:
- **Small Dataset** (1,000 samples, 64 dimensions): 5-10 minutes
- **Medium Dataset** (10,000 samples, 64 dimensions): 30-60 minutes
- **Large Dataset** (50,000+ samples, 64 dimensions): 2-4 hours

**Bottlenecks Identified**:
1. **Temporal Pattern Storage**: 24x memory multiplication
2. **LOGO Cross-Validation**: s-fold increase in regression time
3. **Interactive Plot Generation**: Plotly HTML rendering
4. **PCA on Large Matrices**: O(d³) component for d×d covariance

**Quick Mode Optimizations**:
- Reduced heatmap samples (50 instead of 100)
- Fewer worker threads (4 instead of 8)
- Skip computationally expensive components
- Smaller PCA component count

### Limitations and Known Issues

#### Statistical Limitations

**Multiple Comparison Problem**:
- **Issue**: Many statistical tests performed without correction
- **Impact**: Inflated Type I error rate
- **Workaround**: Interpret p-values conservatively
- **Not Implemented**: Bonferroni, FDR correction

**Assumption Violations**:
- **Independence**: Cage effects not modeled (animals from same cage)
- **Normality**: Embedding distributions not tested
- **Linearity**: Non-linear age relationships not explored
- **Homoscedasticity**: Constant variance not verified

**Sample Size Requirements**:
- **Strain Analysis**: Requires minimum 5 samples per strain
- **Sliding Windows**: Minimum 20 samples per window
- **PCA**: Stable results need n >> d (samples >> dimensions)
- **Clustering**: Quality depends on sufficient cluster sizes

#### Technical Limitations

**Data Format Requirements**:
- **Embeddings**: Must be in specific dictionary format
- **Metadata**: Required columns ('avg_age_days_chunk_start', 'strain')
- **Temporal**: Assumes exactly 1440 frames per sequence
- **Keys**: Frame mapping requires specific "CAGE_ID_DATE" format

**Scalability Issues**:
- **Memory**: Temporal analysis requires 24x memory overhead
- **Time**: LOGO cross-validation scales linearly with strain count
- **Storage**: Large datasets produce numerous large output files
- **Interactive**: Plotly plots become slow with >10,000 points

**Feature Gaps**:
- **Causal Inference**: No causal modeling of age/behavior relationships
- **Hierarchical Models**: Cage/litter effects not modeled
- **Time Series**: No temporal autocorrelation analysis
- **Non-linear**: Limited non-linear modeling (only Random Forest)
- **Uncertainty**: No confidence intervals on predictions

### Interpretation Guidelines

#### Regression Analysis Results

**RMSE Interpretation**:
- **Units**: Same as target variable (days for age prediction)
- **Baseline Comparison**: Compare to baseline MAE (mean prediction)
- **Strain Variation**: Higher RMSE may indicate strain-specific aging patterns
- **Dimension Ranking**: Lower RMSE indicates better age-predictive dimensions

**R² Interpretation**:
- **Range**: -∞ to 1.0 (negative indicates worse than mean prediction)
- **Biological Significance**: R² > 0.1 suggests meaningful relationship
- **Dimension Comparison**: Relative ranking more important than absolute values
- **Cross-Validation**: LOGO R² typically lower than train/test split

#### Feature Importance Results

**Univariate Correlations**:
- **Magnitude**: |r| > 0.3 suggests strong linear relationship
- **Significance**: p < 0.05 with caution for multiple comparisons
- **Direction**: Sign indicates positive/negative age correlation

**Consensus Ranking**:
- **Robust**: Average across methods reduces method-specific biases
- **Top 10**: Focus on most consistently important dimensions
- **Stability**: Rankings may vary with different train/test splits

#### Clustering Analysis Results

**Silhouette Scores**:
- **Range**: -1 to 1 (higher is better)
- **Interpretation**: >0.5 good, >0.7 excellent cluster separation
- **Algorithm Comparison**: Compare across methods for best approach
- **Biological Relevance**: High silhouette doesn't guarantee biological meaning

**Cluster Characteristics**:
- **Age Distribution**: Look for age-stratified vs. age-mixed clusters
- **Strain Enrichment**: Pure strain clusters suggest genetic effects
- **Size Balance**: Very small clusters may be outlier groups

#### Anomaly Detection Results

**Distance Threshold**:
- **Statistical**: 2.5 standard deviations is conservative threshold
- **Biological**: Consider experimental procedures on detected dates
- **Validation**: Manual inspection of anomalous samples recommended

**Common Anomaly Sources**:
- **Experimental Procedures**: Phenotyping, handling, environmental changes
- **Technical Issues**: Recording problems, calibration errors
- **Biological Variation**: Illness, unusual behavior, developmental stages

### Troubleshooting Guide

#### Common Error Messages

**"Embeddings file not found"**:
- Check file path spelling and existence
- Ensure .npy file extension
- Verify file permissions

**"Required label 'avg_age_days_chunk_start' not found"**:
- Check metadata CSV column names
- Remove extra spaces from column headers
- Ensure consistent data types (numeric for age)

**"No valid data remains after removing NaNs"**:
- Check for missing values in age/strain columns
- Verify embedding data quality (no NaN values)
- Consider increasing sample size

**Memory errors during temporal analysis**:
- Use --quick-mode flag
- Reduce --max-workers parameter
- Monitor system memory usage
- Consider running basic analysis only

**"GAM fitting failed" (bodyweight integration)**:
- Switch to --bodyweight-strategy closest
- Check for sufficient bodyweight measurements per animal
- Verify date formats in bodyweight files

#### Performance Optimization

**For Large Datasets (>50,000 samples)**:
1. Use --quick-mode for initial exploration
2. Reduce --n-heatmap-samples to 50
3. Set --max-workers to number of CPU cores
4. Skip interactive plots (--skip-interactive)
5. Run basic and enhanced analysis separately

**For Limited Memory Systems (<16GB RAM)**:
1. Use --skip-temporal to avoid 24x memory multiplication
2. Process subsets of data separately
3. Monitor memory usage during PCA analysis
4. Consider dimension reduction before analysis

**For Faster Iteration**:
1. Use --quick-mode for parameter testing
2. Skip computationally expensive components
3. Focus on specific analysis types (--analysis-type basic)
4. Use smaller sample sizes for visualization

### Extension and Customization

#### Adding New Analysis Methods

**Regression Models**:
```python
# Add to models dictionary in perform_full_model_regression()
models = {
    'Ridge': Ridge(alpha=1.0),
    'Linear': LinearRegression(),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),  # Add new model
    'SVR': SVR(kernel='rbf'),  # Add SVM regression
}
```

**Feature Selection Methods**:
```python
# Add to selection_methods in perform_feature_selection_regression()
selection_methods = {
    'SelectKBest_10': SelectKBest(score_func=f_regression, k=10),
    'Mutual_Info': SelectKBest(score_func=mutual_info_regression, k=10),  # Add mutual information
    'Boruta': BorutaPy(RandomForestRegressor(), random_state=42),  # Add Boruta (requires installation)
}
```

**Clustering Algorithms**:
```python
# Add to clustering_algorithms in perform_clustering_analysis()
clustering_algorithms = {
    'KMeans_5': KMeans(n_clusters=5, random_state=42),
    'SpectralClustering': SpectralClustering(n_clusters=5, random_state=42),  # Add spectral
    'GaussianMixture': GaussianMixture(n_components=5, random_state=42),  # Add GMM
}
```

#### Parameter Customization

**Model Parameters**:
- Ridge alpha: Increase for more regularization (default: 1.0)
- PCA components: Adjust based on explained variance needs
- Cluster counts: Modify based on expected biological groups
- Window sizes: Adjust for different temporal resolutions

**Visualization Parameters**:
- Figure sizes: Modify figsize parameters for different display requirements
- Color schemes: Change cmap parameters for different color palettes
- DPI settings: Adjust for different print/display requirements
- Interactive plot sizes: Modify width/height for different screen sizes

#### Adding New Metadata Variables

**Continuous Variables** (e.g., body length, temperature):
```python
# Add to correlation analysis
if 'body_length' in self.metadata_aligned.columns:
    correlation_data['body_length'] = self.metadata_aligned['body_length'].values

# Add to regression as covariate
if include_body_length and 'body_length' in self.metadata_aligned.columns:
    body_length = self.metadata_aligned['body_length'].values.reshape(-1, 1)
    X = np.concatenate([X, body_length], axis=1)
```

**Categorical Variables** (e.g., sex, treatment):
```python
# Add to classification analysis
if 'sex' in self.metadata_aligned.columns:
    y_sex = self.metadata_aligned['sex'].values
    # Perform sex classification similar to strain classification
```

### Research Applications and Use Cases

#### Aging Research Applications

**Biomarker Discovery**:
- Identify embedding dimensions most predictive of age
- Compare age-predictive patterns across strains
- Discover strain-specific aging signatures
- Validate embedding-based age prediction against known biomarkers

**Temporal Pattern Analysis**:
- Characterize circadian rhythm changes with age
- Identify age-related changes in activity patterns
- Discover behavioral patterns that precede age-related phenotypes
- Compare temporal stability across different age ranges

**Strain Comparison Studies**:
- Identify genetic effects on behavioral aging
- Discover strain-specific temporal patterns
- Validate embedding-based strain classification
- Map behavioral differences to genetic variants

#### Experimental Design Applications

**Power Analysis**:
- Determine sample sizes needed for age prediction accuracy
- Identify most informative dimensions for future studies
- Optimize experimental design based on embedding variance
- Plan longitudinal studies based on temporal pattern stability

**Quality Control**:
- Detect experimental anomalies through outlier analysis
- Identify batch effects through temporal clustering
- Validate experimental procedures through pattern consistency
- Monitor data quality through embedding stability

**Phenotype Discovery**:
- Discover novel behavioral phenotypes through clustering
- Identify rare behavioral variants through anomaly detection
- Characterize behavioral subgroups within strains
- Map behavioral space through dimensionality reduction

#### Therapeutic Research Applications

**Drug Effect Assessment**:
- Compare treated vs. untreated embedding patterns
- Identify drug-responsive behavioral dimensions
- Characterize dose-response relationships in embedding space
- Discover unexpected drug effects through pattern analysis

**Disease Model Validation**:
- Compare disease model embedding patterns to wild-type
- Identify disease-specific behavioral signatures
- Validate therapeutic interventions through pattern normalization
- Characterize disease progression through temporal analysis

### Technical Architecture Details

#### Design Patterns Used

**Factory Pattern**:
- Model creation in regression analysis
- Feature selection method instantiation
- Clustering algorithm selection

**Strategy Pattern**:
- Bodyweight matching strategies
- Analysis type selection (basic vs. enhanced)
- Validation method selection (LOGO vs. train/test)

**Observer Pattern**:
- Progress tracking through tqdm
- Result collection across parallel processes
- Status updates during long-running analyses

**Template Method Pattern**:
- Analysis pipeline structure
- Common preprocessing steps
- Standardized output generation

#### Error Handling Strategy

**Graceful Degradation**:
- Continue analysis if single component fails
- Provide meaningful error messages
- Offer alternative analysis paths
- Save partial results before failures

**Input Validation**:
- File existence and format checking
- Data type and range validation
- Column presence verification
- Memory requirement estimation

**Resource Management**:
- Explicit garbage collection after large operations
- File handle cleanup in exception cases
- Thread pool proper shutdown
- Temporary file cleanup

#### Testing and Validation

**Unit Tests** (not included but recommended):
- Individual function correctness
- Edge case handling
- Parameter validation
- Data format compatibility

**Integration Tests**:
- End-to-end pipeline functionality
- Cross-platform compatibility
- Memory usage validation
- Performance benchmarking

**Validation Studies**:
- Compare results against manual analysis
- Validate against published datasets
- Cross-validate with different random seeds
- Reproduce results across different systems

### Future Development Roadmap

#### Planned Enhancements

**Statistical Improvements**:
- Multiple comparison correction
- Confidence interval estimation
- Bootstrap resampling for robustness
- Permutation testing for significance

**Model Additions**:
- Deep learning models (neural networks)
- Time series specific models (LSTM, ARIMA)
- Hierarchical models (mixed effects)
- Causal inference methods

**Visualization Enhancements**:
- Real-time interactive dashboards
- Animation through time/age
- Custom color scheme selection
- Publication-ready automated formatting

**Performance Optimizations**:
- GPU acceleration for large computations
- Distributed computing support
- Memory-mapped file operations
- Incremental analysis for large datasets

#### Research Extensions

**Multi-modal Integration**:
- Combine with physiological data
- Integrate with genomic information
- Include environmental measurements
- Correlate with neural activity data

**Advanced Temporal Analysis**:
- Hidden Markov models for state transitions
- Change point detection algorithms
- Seasonal decomposition methods
- Long-term trend analysis

**Causal Analysis**:
- Mendelian randomization approaches
- Instrumental variable methods
- Directed acyclic graph modeling
- Counterfactual analysis

### Citation and References

#### Academic References

**Machine Learning Methods**:
- Scikit-learn: Pedregosa et al. (2011) JMLR
- Ridge Regression: Hoerl & Kennard (1970) Technometrics
- Random Forest: Breiman (2001) Machine Learning
- PCA: Hotelling (1933) Journal of Educational Psychology

**Statistical Methods**:
- Leave-One-Group-Out CV: Arlot & Celisse (2010) Statistics Surveys
- Silhouette Analysis: Rousseeuw (1987) Journal of Computational and Applied Mathematics
- GAM: Hastie & Tibshirani (1990) Generalized Additive Models

**Visualization Libraries**:
- Matplotlib: Hunter (2007) Computing in Science & Engineering
- Seaborn: Waskom et al. (2017) JOSS
- Plotly: Plotly Technologies Inc. (2015)

#### Behavioral Embedding References

**hBehaveMAE Architecture**:
- Masked Autoencoders: He et al. (2022) CVPR
- Vision Transformer: Dosovitskiy et al. (2021) ICLR
- Behavioral Analysis: [Cite relevant behavioral analysis papers]

**Circadian Analysis**:
- Circadian Rhythm Analysis: [Cite relevant circadian papers]
- Temporal Pattern Recognition: [Cite relevant temporal analysis papers]

### License and Usage Terms

#### Software License

This analysis framework is provided as research software. Users should:

1. **Cite appropriately** when using in publications
2. **Acknowledge dependencies** (scikit-learn, matplotlib, etc.)
3. **Share modifications** that improve the framework
4. **Report issues** to help improve the software

#### Data Usage Guidelines

**Privacy Considerations**:
- Animal data should follow institutional guidelines
- Remove identifying information when sharing results
- Follow data sharing agreements and protocols
- Respect intellectual property of dataset creators

**Reproducibility Standards**:
- Save analysis parameters and random seeds
- Document data preprocessing steps
- Archive analysis software versions
- Share processed data when permitted

#### Disclaimer

This software is provided "as is" without warranty. Users are responsible for:
- Validating results for their specific use case
- Understanding statistical assumptions and limitations
- Following appropriate experimental design principles
- Interpreting results within biological context

The authors are not responsible for misinterpretation or misuse of analysis results.

---

## Quick Start Example

For immediate use with your data:

```bash
# Clone or download the analysis scripts
# Ensure your embeddings file is in the expected format
# Prepare metadata CSV with required columns

# Run basic analysis
python run_analysis.py /path/to/embeddings.npy /path/to/metadata.csv

# Run with bodyweight integration
python run_analysis.py /path/to/embeddings.npy /path/to/metadata.csv \
  --bodyweight-file /path/to/BW.csv \
  --tissue-file /path/to/TissueCollection_PP.csv \
  --summary-file /path/to/summary_metadata.csv

# Quick test run
python run_analysis.py /path/to/embeddings.npy /path/to/metadata.csv --quick-mode
```

Results will be saved in `embedding_analysis_results/` directory with comprehensive outputs for publication and further analysis.