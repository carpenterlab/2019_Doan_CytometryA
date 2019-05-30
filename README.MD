# Label-free tracking of acute lymphoblastic leukemia by deep learning
Analytic scripts for Doan et al. (2018), "Label-free tracking of acute lymphoblastic leukemia by computer vision"

# Usage:

- Step 0: IDEAS 6.2 - Preliminary gating, remove out-of-focus, collect single cells, and exporting .CIF
- Step 1: Python 3.6 - Parse little images inside .CIF into .NPY
- Step 2: Python 3.6 - Train convolutional neural network ResNet50
- Step 3: Python 3.6 - Evaluate trained model, supervised classification of RBC morphology
- Step 3b: Python 3.6 - Data-driven visualization of deep learning feature space
- Step 3c: Python 3.6 - Identify leukemic blast in a mixture of White blood cells

iPython notebooks for steps 1-3 are provided in this repository

# Dependencies:
Prior to installation of deepometry itself, user needs the following packages pre-installed:

- python 3.6.3
- h5py==2.8.0
- javabridge==1.0.17
- Keras==2.1.15
- keras-resnet==0.0.7
- matplotlib==2.2.2
- numpy==1.14.5
- opencv-python==3.4.1.15
- pandas==0.20.3
- Pillow==5.1.0
- python-bioformats==1.4.0
- scikit-image==0.14.0
- scikit-learn==0.19.1
- scipy==1.1.0
- seaborn==0.8.1
- tensorboard==1.9.0
- tensorflow-gpu==1.9.0rc1

Note: Java development kit (32- or 64- bit version to be matched with operating system) should be installed before python-bioformats and javabridge.

Note: Tensorflow python package is sufficient for CPU use. However, in order to utilize a CUDA-compatible GPU, Tensorflow-GPU as well as CUDA and cuDNN packages are required; more details are described on Tensorflow homepage.

Note: Windows user will need Microsoft Visual C++ Build tools and its compilers installed, with respect to Python versions of 2.7 or 3.5+ accordingly. Windows user is also advised to install numpy (numpy+mkl version), scipy and scikit-image as wheel packages: numpy-1.12.1+mkl-cp35-cp35m-win_amd64.whl, scipy-0.19.0-cp35-cp35m-win_amd64.whl, scikit_image-0.13.0-cp35-cp35m-win_amd64.whl

# Source code and development of deepometry: 

Please visit http://github.com/broadinstitute/deepometry
