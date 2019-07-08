# Label-free monitoring of acute lymphoblastic leukemia by computer vision

Analytic scripts for Doan et al. (2019), "Label-free tracking of acute lymphoblastic leukemia by computer vision"

Throughout this pipeline, imaging flow cytometry single-cell images (.CIF format) are resized to 48 x 48 pixels by cropping the peripheral background or padding channel-wise with randomly distributed noise sampled from the background. Additionally, cell images were contrast-stretched channel-wise to rescale the intensities between the 0.5 and 99.5 percentiles to the full range of uint8, [0, 256). We adopted ResNet architecture using a Python framework [Keras-resnet](https://github.com/broadinstitute/keras-resnet). The network includes 50 convolutional layers, forming repetitive blocks that perform residual learning, followed by fully connected and softmax layers.

We computed categorical cross-entropy as the loss function and accuracy as our metric, respectively. The model was compiled using the Adam optimizer with a learning rate of 0.0001. The learning rate was reduced by a factor of 10 when the validation loss failed to improve for 10 consecutive epochs. Training was set to stop after 25 consecutive epochs of no improvement in the validation loss.

Objects were categorized as “leukemic”, “normal” (not leukemic), and “other” (non-lymphoid nucleated cells such as granulocytes, monocytes, dead or deformed cells). Training and validation data was randomly undersampled per-patient across cell type to create a balanced data set. 80% of sampled data was assigned to the training data set, with the remaining 20% assigned to validation. 

The data was zero-centered using channel-wise mean subtraction. Means were precomputed from the training set. Mean subtraction and augmentation were performed in real time during training and validating operations. Augmentation included random combinations of horizontal or vertical flips, horizontal or vertical shifts (up to 50% of the image size), and rotations up to 180 degrees.

Augmented training and validation data were generated in batches of 256 images to maximize GPU memory resources. We configured the model to train for a maximum of 512 epochs, though early stopping generally terminated training before 200 epochs. Each epoch ran M / 256 steps, with M as the number of training samples, to ensure the entire training set was seen once per epoch. Validation occurred once at the end of each epoch, using the entire validation set with validation step K / 256, where K is the number of validation samples.

Test data was comprised entirely of withheld patient data. Before prediction or evaluation, the mean pixel values obtained from the training datasets were subtracted from the test data. No other processing or augmentation was applied.

# Sample metadata

## Patients ##

Bone marrow samples from pediatric patients (less than 18 years) who presented with B lineage ALL within the northern region of England (October 2013 to July 2015) were used in this study. They were obtained from the Newcastle Haematology Biobank following project approval (reference 2002/11 and 07/H906). The children were registered on the UKALL2011 protocol and were treated with 3-4 drugs during the induction phase of treatment when on-treatment samples were assessed.

## Data acquisition ##

Bone marrow samples were lysed with Lyse/Fix 5x Buffer (BD Biosciences) and labeled with CD45-APCH7, CD10-PE, CD34-PE Texas Red, CD19-APC, DAPI and p65-FITC. The latter was included as an antigen not recognised to play a part in the LAIP. IFC measurement was conducted using a dual camera ImageStream X MarkII system (Amnis, Seattle, USA), equipped with 488nm, 405nm, 561nm and 642nm excitation lasers. Data was collected at 40× magnification, pixel size 0.3 x 0.3 µm. Bright-field illumination was collected in channels 1 (camera 1, 430nm–480nm) and 9 (camera 2, 570 nm–595 nm). Dark-field illumination was collected in channel 6 (camera 1, 745 nm – 800 nm) from a 758 nm laser source. Emission from CD10-PE was measured from the 488 nm laser in channel 3 (560 nm–595 nm), Texas Red emission was measured from the 488 nm laser in channel 4 (595 nm–660 nm), CD19-APC from the 642 nm laser in channel 11 (660 nm–745 nm), CD45-APCH7 from the 654 nm laser in channel 12 (740 nm–800 nm) and finally DAPI emission was measured from the 405 nm laser in channel 7 (430 nm–505 nm). Standard flow cytometric compensation procedure was applied in each sample.

Data analysis first started with the exclusion of clumped cells and out-of-focus cells based on aspect ratios, size and gradient root mean square of typical non-cellular events. We then constructed pairwise 2-D scatter plots and performed the manual sequential gating to identify ALL cells as for the standardized flow method. In addition, we identified normal B cells (CD19+, CD34-, CD45+, and CD10+/-) and classified cells with high side scatter, CD19+ and DAPI+ as ‘other’; while DAPI negative events were classified as red cell/debris. Images from fluorescent, bright-field, and dark-field channels of gated cell populations were exported into a file container (.CIF).

|                | Bright -field 1 | p65 -FITC | CD10 -PE | CD34 -TexaRed | Unused ch.1 | Dark-field | DAPI  | Unused ch.2 | Bright -field 2 | Unused ch.3 | CD19 -APC | CD45 -APCH7 |
|----------------|-----------------|-----------|----------|---------------|-------------|------------|-------|-------------|-----------------|-------------|-----------|-------------|
| Bright-field 1 | 1               | 0.032     | 0.033    | 0.051         | 0           | 0          | 0.046 | 0           | 0               | 0           | 0.001     | 0.001       |
| p65-FITC       | 0.031           | 1         | 0.102    | 0.055         | 0           | 0          | 0.048 | 0           | 0               | 0           | 0.001     | 0.001       |
| CD10-PE        | 0               | 0.197     | 1        | 0.28          | 0           | 0          | 0.01  | 0           | 0               | 0           | 0.001     | 0.001       |
| CD34-TexaRed   | 0               | 0.052     | 0.313    | 1             | 0           | 0          | 0.003 | 0           | 0               | 0           | 0.001     | 0.001       |
| Unused ch.1    | 0               | 0.018     | 0        | 0             | 1           | 0          | 0.003 | 0           | 0               | 0           | 0.018     | 0.002       |
| Dark-field     | 0.017           | 0.022     | 0.036    | 0.089         | 0           | 1          | 0.003 | 0           | 0               | 0           | 0.002     | 0.021       |
| DAPI           | 0.025           | 0.002     | 0.002    | 0.005         | 0           | 0          | 1     | 0           | 0.012           | 0           | 0.025     | 0.003       |
| Unused ch.2    | 0               | 0         | 0        | 0             | 0           | 0          | 0     | 1           | 0.011           | 0           | 0         | 0           |
| Bright-field 2 | 0               | 0.006     | 0.053    | 0.014         | 0           | 0          | 0.125 | 0           | 1               | 0           | 0.024     | 0.002       |
| Unused ch.3    | 0               | 0.001     | 0.016    | 0             | 0           | 0          | 0.049 | 0           | 0.038           | 1           | 0.024     | 0.002       |
| CD19-APC       | 0               | 0.003     | 0.006    | 0.035         | 0           | 0          | 0.023 | 0           | 0.013           | 0           | 1         | 0.058       |
| CD45-APCH7     | 0               | 0         | 0.002    | 0.006         | 0           | 0          | 0.04  | 0           | 0.015           | 0           | 0.148     | 1           |

Table: Compensation matrix used in imaging flow cytometry measurement with ImageStream X MarkII in this study.

# Usage

- Step 0: IDEAS 6.2 - Preliminary gating, remove out-of-focus, collect single cells, and exporting .CIF
- Step 1: Python 3.7 - Parse little images inside .CIF into .NPY
- Step 2: Python 3.7 - Train convolutional neural network ResNet50
- Step 3: Python 3.7 - Evaluate trained model, supervised classification of RBC morphology
- Step 3b: Python 3.7 - Data-driven visualization of deep learning feature space
- Step 3c: Python 3.7 - Identify leukemic blast in a mixture of White blood cells

Only steps 1-3 are included in this Github repository

# Dependencies

- python 3.6+
- h5py==2.8.0
- javabridge==1.0.17
- keras==2.1.15
- keras-resnet==0.0.7
- matplotlib==2.2.2
- numpy==1.14.5
- opencv-python==3.4.1.15
- pandas==0.20.3
- pillow==5.1.0
- python-bioformats==1.4.0
- scikit-image==0.14.0
- scikit-learn==0.19.1
- scipy==1.1.0
- seaborn==0.8.1
- tensorboard==1.9.0
- tensorflow-gpu==1.9.0rc1

Note 1: [Java Development Kit](https://www.oracle.com/technetwork/java/javase/downloads/jdk11-downloads-5066655.html) should be installed before python-bioformats and javabridge. JDK 8.0 or 11.0 is acceptable but *NOT* 12. The choice of JDK 32 or 64 bit version should match with operating system.

Note 2: In order to utilize a CUDA-compatible GPU, Tensorflow-GPU, CUDA, as well as cuDNN packages are required. More details are described on [Tensorflow homepage](https://www.tensorflow.org/install). Although non-GPU Tensorflow (CPU only) is compatible with this pipeline, it is less efficient in term of speed.

Note 3: Windows users may need to install:
- [Visual Studio](https://visualstudio.microsoft.com/) (Community or better, 2017 or later), and its standard "Desktop development with C++" package
- [VS Build tools](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16) (2017 or later), and its standard "Visual C++ build tools" package
- (Optional) [Conda](https://www.anaconda.com/distribution/) is recommended but not obligatory.

# Comparative workflow

## Conventional image analytics ##

Images contained within a .CIF file were stitched into montages by using a Python [stitching script](https://github.com/CellProfiler/stitching). Cellular objects from the montages were identified (segmented) using [CellProfiler](https://cellprofiler.org/) 3.1. Object features were extracted by a series of built-in measurement modules, including measuring object intensity, size, shapes, textures, correlations, and subcellular components. Data cleaning and feature selection were performed by [Cytominer](https://github.com/cytomining/cytominer/) to remove features with near-zero variance and features that have poor correlation across replicates. Redundant features that are highly correlated were then identified and only one feature for each of these groups was retained. After pruning, no pair of features had a correlation greater than the 95% cut-off threshold.

## Conventional machine learning ##
Various machine learning algorithms were tested and their hyperparameters were optimized by [Hyperopt](https://github.com/hyperopt/hyperopt)**, including naive Bayes, random forest and support vector machine (SVM). We eventually chose linear SVM as the algorithm of choice for classical machine learning to achieve an acceptable balance between performance and computational efficiency. We trained the classifier to differentiate ALL cells from normal B lymphocytes with different combinations of antibody and DNA biomarkers. In parallel, we iterated the training-testing sets on 20 datasets (leave-one-instance-out) to observe the variance of prediction accuracy due to the clinical diversity of patients.

** Bergstra, J., Komer, B., Eliasmith, C., Yamins, D. & Cox, D. D. Hyperopt: a Python library for model selection and hyperparameter optimization. Comput. Sci. Discov. 8, 014008 (2015).

# Source code and development of deepometry: 

Please visit http://github.com/broadinstitute/deepometry
