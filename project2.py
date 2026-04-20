from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from scipy.signal import stft, istft
from encodec import EncodecModel
from encodec.utils import convert_audio

from metrics import snr_db, mse, compression_ratio
# CONFIG
INPUT = Path("Perfect_Alignment.mp3")
OUT_DIR = Path("output_2")
TABLE_DIR = Path("results_2")

OUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

BANDWIDTHS = [1.5, 3.0, 6.0, 12.0]
# SEPARATION
def separate(audio, sr):
    f, t, Zxx = stft(audio, sr, nperseg=1024)
    mag = np.abs(Zxx)
    bass = np.zeros_like(mag)
    vocal = np.zeros_like(mag)
    high = np.zeros_like(mag)
    for i, freq in enumerate(f):
        if freq < 200:
            bass[i] = mag[i]
        elif 300 <= freq <= 3000:
            vocal[i] = mag[i]
        else:
            high[i] = mag[i]
    total = bass + vocal + high + 1e-9
    Z_bass = Zxx * (bass / total)
    Z_vocal = Zxx * (vocal / total)
    Z_high = Zxx * (high / total)
    _, bass_audio = istft(Z_bass, sr)
    _, vocal_audio = istft(Z_vocal, sr)
    _, high_audio = istft(Z_high, sr)
    return bass_audio, vocal_audio, high_audio
# ENCODEC PROCESS (1 STEM)
def encodec_process(audio_np, sr, model, name, OUT_DIR):
    audio_np=audio_np.astype(np.float32)
    wav = torch.tensor(audio_np).unsqueeze(0)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)
    with torch.no_grad():
        encoded = model.encode(wav)
        encoded_path = OUT_DIR / f"{name}_encoded.pt"
        torch.save(encoded, encoded_path)
    with torch.no_grad():
        decoded = model.decode(encoded)
    rec = decoded.squeeze(0).cpu().numpy()
    rec = np.squeeze(rec)
    rec = rec.astype(np.float32)
    rec = rec / (np.max(np.abs(rec)) + 1e-9)
    wav_path = OUT_DIR / f"{name}_compressed.wav"
    sf.write(wav_path, rec, model.sample_rate)
    return rec, encoded_path
def calc_bitrate(file_path, duration):
    size_bits = file_path.stat().st_size * 8
    return size_bits / duration / 1000  # kbps
# MAIN PIPELINE
def main():
    audio, sr = sf.read(INPUT)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    # SEPARATE
    bass, vocal, high = separate(audio, sr)
    duration_sec = len(audio) / sr
    original_bits = len(audio) * 16  # PCM 16-bit mono
    # normalize original
    audio = audio / np.max(np.abs(audio))
    model = EncodecModel.encodec_model_24khz()
    model.eval()
    rows = []
    # LOOP BITRATE
    for bw in BANDWIDTHS:
        model.set_target_bandwidth(bw)
        # compress từng stem
        bass_rec, bass_file = encodec_process(bass, sr, model, f"bass_{bw}",OUT_DIR)
        vocal_rec, vocal_file = encodec_process(vocal, sr, model, f"vocal_{bw}",OUT_DIR)
        high_rec, high_file = encodec_process(high, sr, model, f"high_{bw}",OUT_DIR)
        # bitrate từng stem
        bass_br = calc_bitrate(bass_file, duration_sec)
        vocal_br = calc_bitrate(vocal_file, duration_sec)
        high_br = calc_bitrate(high_file, duration_sec)
        # align length
        min_len = min(len(bass_rec), len(vocal_rec), len(high_rec))
        # reconstruct
        reconstructed = (bass_rec[:min_len] + vocal_rec[:min_len] + high_rec[:min_len])
        reconstructed = reconstructed / np.max(np.abs(reconstructed) + 1e-9)
        reconstructed = np.squeeze(reconstructed)
        reconstructed = reconstructed.astype(np.float32)
        # save file
        out_path = OUT_DIR / f"reconstructed_{bw:.1f}kbps.wav"
        sf.write(out_path, reconstructed, model.sample_rate)
        # align original
        ref = audio[:min_len]
        # METRICS (YOUR CODE)
        # align length
        min_len = min(len(ref), len(reconstructed))
        ref = ref[:min_len]
        reconstructed = reconstructed[:min_len]
        m = mse(ref, reconstructed)
        s = snr_db(ref, reconstructed)
        compressed_bits = bw * 1000 * duration_sec
        cr = compression_ratio(original_bits, compressed_bits)
        rows.append({
            "bandwidth_kbps": bw,
            "mse": m,
            "snr_db": s,
            "compression_ratio": cr,
            "bass_bitrate": bass_br,
            "vocal_bitrate": vocal_br,
            "high_bitrate": high_br
        })
        print(f"Done: {bw} kbps")
    # SAVE TABLE
    df = pd.DataFrame(rows)
    csv_path = TABLE_DIR / "stem_comparison.csv"
    df.to_csv(csv_path, index=False)
    print("\n===== RESULT =====")
    print(df)
    print(f"\nSaved to {csv_path}")
if __name__ == "__main__":
    main()