import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
import torch
from tqdm import tqdm

from whisper.model import Whisper, ModelDimensions
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram


# ---------------------------
# LOAD MODEL IN FLOAT32 (CPU SAFE)
# ---------------------------
def load_model(path) -> Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location="cpu")
    dims = ModelDimensions(**checkpoint["dims"])
    print(device, dims)

    model = Whisper(dims)

    # Remove decoder (not needed for PPG extraction)
    del model.decoder

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # IMPORTANT: convert whole model to float32 (no fp16!)
    model = model.float()
    for p in model.parameters():
        p.data = p.data.float()

    model.to(device)
    return model


# ---------------------------
# CPU-SAFE PPG EXTRACTION
# ---------------------------
def pred_ppg(whisper: Whisper, wavPath, ppgPath):
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppgln = audln // 320

    audio = pad_or_trim(audio)

    # float32 mel spectrogram
    mel = log_mel_spectrogram(audio).float().to(whisper.device)

    with torch.no_grad():
        # whisper encoder expects 3D tensor [B,80,T]
        mel_input = mel.unsqueeze(0).float()

        # ensure model stays float32
        whisper = whisper.float()

        # forward pass
        ppg = whisper.encoder(mel_input).squeeze().float().cpu().numpy()

        # trim to true length
        ppg = ppg[:ppgln, :]

        # save directory
        os.makedirs(os.path.dirname(ppgPath), exist_ok=True)
        np.save(ppgPath, ppg, allow_pickle=False)


# ---------------------------
# MAIN SCRIPT
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav folder", dest="wav")
    parser.add_argument("-p", "--ppg", help="ppg output folder", dest="ppg")
    args = parser.parse_args()

    print(args.wav)
    print(args.ppg)

    os.makedirs(args.ppg, exist_ok=True)

    wavPath = args.wav
    ppgPath = args.ppg

    # load whisper in float32
    whisper = load_model('/kaggle/input/whisper-large-v2/large-v2.pt')

    # walk through files
    for root, dirs, files in os.walk(wavPath):
        for file in tqdm(files, desc='Processing WAV files'):
            if file.endswith(".wav"):
                relative_path = os.path.relpath(os.path.join(root, file), wavPath)
                path_wav = os.path.join(wavPath, relative_path)

                # output path
                path_ppg = os.path.join(ppgPath, os.path.splitext(relative_path)[0] + "_largev2ppg.npy")

                # skip if exists
                if os.path.isfile(path_ppg):
                    continue

                # generate ppg
                pred_ppg(whisper, path_wav, path_ppg)
