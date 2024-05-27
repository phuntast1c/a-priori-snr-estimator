import argparse
from pathlib import Path

import torch
import torchaudio as ta
from a_priori_snr.models.a_priori_snr_tcn import APrioriSNREstimator
from scipy.io import savemat

# example call:
# python run_model.py --input test.wav --output test.mat --use_freq_subset

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process a .wav file to estimate A-Priori SNR."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input .wav file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Path to save the processed output .mat file",
    )
    parser.add_argument(
        "--use_freq_subset",
        action="store_true",
        help="Use model estimating only {1250, 2250, 3500} Hz",
    )
    return parser.parse_args()


def main():
    """
    Main function for running the a-priori SNR estimator model.

    This function loads the model checkpoint, processes the input waveform, and performs inference to estimate the a-priori SNR.
    The estimated a-priori SNR can be optionally saved to a .mat file.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_ESTIMATING_ALL_FREQS = not args.use_freq_subset
    if MODEL_ESTIMATING_ALL_FREQS:
        checkpoint_path = Path(
            "/home/marvint/dr/code/deep_learning_projects/a-priori-snr-estimator/a_priori_snr/saved/20240524_171010_crazy-block-70/epoch=81-step=738000.ckpt",  # all freqs
        )
    else:
        checkpoint_path = Path(
            "/home/marvint/dr/code/deep_learning_projects/a-priori-snr-estimator/a_priori_snr/saved/20240524_171010_actual-consequence-88/epoch=57-step=522000.ckpt"  # only three freqs
        )

    wav_file_path = Path(args.input)

    model = APrioriSNREstimator.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    model.eval()  # put into evaluation mode (batch normalization handled differently)

    wav_data, fs_wav = ta.load(wav_file_path)
    if wav_data.shape[0] > 1:  # use single channel
        wav_data = wav_data[:1]
    wav_data = wav_data.to(device=device)
    if fs_wav != model.fs:
        print(f"Resampling from {fs_wav} to {model.fs}")
        wav_data = ta.functional.resample(wav_data, orig_freq=fs_wav, new_freq=model.fs)

    wav_data = wav_data.unsqueeze(0)
    inp = {
        "input": wav_data,
    }

    with torch.no_grad():  # no gradient computation needed for inference
        a_priori_snr_estimate = model.perform_inference(inp)[0]

    if args.output:
        # Optionally save the processed output
        savemat(
            args.output,
            {"a_priori_snr_estimate": a_priori_snr_estimate.numpy(force=True)},
        )


if __name__ == "__main__":
    main()
