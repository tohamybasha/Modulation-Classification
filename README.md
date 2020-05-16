# Modulation-Classification

# Dataset 
DeepSig Dataset: RadioML 2016.04C

Download from here : http://opendata.deepsig.io/datasets/2016.10/RML2016.10b.tar.bz2
A synthetic dataset, generated with GNU Radio, consisting of 11 modulations. This is a
variable-SNR dataset with moderate LO drift, light fading, and numerous different
labeled SNR increments for use in measuring performance across different signal and
noise power scenarios.


Every sample is presented using two vectors each of them has 128 elements. You might
try the raw features and you can make a battery of more features such as

1. Raw time series as given (two channels)

2. First derivative in time (two channels)

3. Integral in time (two channels)

4. Combinations of 1, 2 and 3. (More channels)

# Supervised Learning Step
-Split the data into 50% for training/validation and 50% for testing.

-We will use a Fully Connected Neural Net as a baseline. You will need
to tune the parameters for best performance. For simplicity use ReLU and
ADAM with default parameters. Ensure you change number of layers and
number of units. Use Early stopping on a validation set of size 5% of the
data.

-We will apply the CNN architecture shown below. The number of
channels in the input layer might be changed as you apply different types
of features.

![CNN Arch](https://user-images.githubusercontent.com/27583722/82118156-8a91cc00-9775-11ea-90a9-b75abf50a638.png)


Results can be found in PDF file attached.


