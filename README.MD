# Label-free tracking of acute lymphoblastic leukemia by deep learning
Analytic scripts for Doan et al. (2018), "Label-free tracking of acute lymphoblastic leukemia by computer vision"

Imaging flow cytometry single-cell images were exported from .CIF files. Data from fluorescent, bright-field, and dark-field channels were exported. The images were resized to 48 x 48 pixels by cropping the peripheral background or padding channel-wise with randomly distributed noise sampled from the background. Additionally, cell images were contrast-stretched channel-wise to rescale the intensities between the 0.5 and 99.5 percentiles to the full range of uint8, [0, 256). We adopted ResNet architecture using a Python framework https://github.com/broadinstitute/keras-resnet. The network includes 50 convolutional layers, forming repetitive blocks that perform residual learning, followed by fully connected and softmax layers.

We computed categorical cross-entropy as the loss function and accuracy as our metric, respectively. The model was compiled using the Adam optimizer with a learning rate of 0.0001. The learning rate was reduced by a factor of 10 when the validation loss failed to improve for 10 consecutive epochs. Training was set to stop after 25 consecutive epochs of no improvement in the validation loss.

Objects were categorized as “leukemic”, “normal” (not leukemic), and “other” (non-lymphoid nucleated cells such as granulocytes, monocytes, dead or deformed cells). Training and validation data was randomly undersampled per-patient across cell type to create a balanced data set. 80% of sampled data was assigned to the training data set, with the remaining 20% assigned to validation. 

The data was zero-centered using channel-wise mean subtraction. Means were precomputed from the training set. Mean subtraction and augmentation were performed in real time during training and validating operations. Augmentation included random combinations of horizontal or vertical flips, horizontal or vertical shifts (up to 50% of the image size), and rotations up to 180 degrees.

Augmented training and validation data were generated in batches of 256 images to maximize GPU memory resources. We configured the model to train for a maximum of 512 epochs, though early stopping generally terminated training before 200 epochs. Each epoch ran M / 256 steps, with M as the number of training samples, to ensure the entire training set was seen once per epoch. Validation occurred once at the end of each epoch, using the entire validation set with validation step K / 256, where K is the number of validation samples.

Test data was comprised entirely of withheld patient data. Before prediction or evaluation, the mean pixel values obtained from the training datasets were subtracted from the test data. No other processing or augmentation was applied.

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
