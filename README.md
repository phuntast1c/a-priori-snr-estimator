# Temporal Convolutional Network-Based SNR Estimation
## Description
This repository contains the code to train and test a temporal convolutional network (TCN)-based a-priori signal-to-noise ratio (SNR) estimator.
The TCN maps log-magnitude and phase features to an estimate of the compressed SNR in [0, 1].
During training, the compressed SNR according to Eq. (11) in [[1]](#1) is used.
During inference, the compressed SNR is expanded as in Eq. (12).
The mean and variance of the SNR in dB required in Eqs. (11) and (12) were estimated using 4000 randomly chosen utterances from the training dataset.

## How to run
### Installation
First, install dependencies:

```bash
# clone project
git clone https://github.com/phuntast1c/a-priori-snr-estimator

# install and activate conda environment
cd a-priori-snr-estimator
conda env create -f environment.yml
conda activate a_priori_snr_estimator
```

### Usage
This repository contains two pre-trained models, one estimating the SNR for all frequency bins and one only estimating the SNR for {1250, 2250, 3500} Hz (fs = 16 kHz).
These models were trained on the reverberant ICASSP 2021 Deep Noise Suppression (DNS) challenge dataset, including both simulated and measured room impulse responses.
```inference.ipynb``` demonstrates how to use these models for the prediction of the SNR given a noisy single-channel input.
```run_model.py``` can be used to save the estimated SNR to a .mat file, e.g. as:

```bash
python run_model.py --input test.wav --output test.mat
```
or, for the second mentioned model, as:

```bash
python run_model.py --input test.wav --output test.mat --use_freq_subset
```

When different settings or a different dataset are desired, the SNR estimator can be trained using the PyTorch Lightning (PL) command-line interface, preferably with an equipped NVIDIA GPU. The available model arguments can be printed using:

```bash
python cli.py fit --model.help APrioriSNREstimator
```

For example, the provided models were trained using:

```bash
python cli.py fit --trainer=configs/trainer/trainer.yaml --model=configs/model/240523_a_priori_snr_tcn.yaml --data=configs/data/240206_dns2_reverberant.yaml --model.limit_frequencies=false --model.use_batchnorm=true --model.layer=8 --model.stack=3 &> /dev/null &
python cli.py fit --trainer=configs/trainer/trainer.yaml --model=configs/model/240523_a_priori_snr_tcn.yaml --data=configs/data/240206_dns2_reverberant.yaml --model.limit_frequencies=true --model.use_batchnorm=true --model.layer=8 --model.stack=3 &> /dev/null &
```

To handle data from the DNS challenge , this repository includes a `PL LightningDataModule` implementation. For instructions on how to obtain the data, please refer to [the official repository](https://github.com/microsoft/DNS-Challenge), and adjust the paths accordingly. The configuration file used to generate the training dataset can be found in `spp/datasets/noisyspeech_synthesizer.cfg`.

## Reference
<a id="1">[1]</a>
Zhang, Q., Nicolson, A. M., Wang, M., Paliwal, K. & Wang, C.-X. DeepMMSE: A Deep Learning Approach to MMSE-based Noise Power Spectral Density Estimation. IEEE/ACM Trans. Audio, Speech and Lang. Proc. 28, 1404–1415 (2020).
