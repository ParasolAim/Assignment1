from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
import soundfile as sf
from scipy.signal import stft, istft
from encodec import EncodecModel
from encodec.utils import convert_audio

# Giả định các hàm metrics của bạn nằm trong file metrics.py ở cùng thư mục
try:
    from metrics import snr_db, mse, compression_ratio
except ImportError:
    # Định nghĩa tạm nếu chưa có file metrics để code không bị crash
    def mse(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
    def snr_db(y_true, y_pred):
        power_true = np.mean(y_true ** 2) + 1e-9
        power_noise = np.mean((y_true - y_pred) ** 2) + 1e-9
        return 10 * np.log10(power_true / power_noise)
    def compression_ratio(orig, comp): return orig / (comp + 1e-9)

# CONFIG THƯ MỤC
INPUT_DIR = Path("test_audios")   # Thư mục chứa các file nhạc test (.mp3, .wav)
OUT_DIR = Path("output_2")         # Thư mục lưu các file stem và file tổng sau nén
TABLE_DIR = Path("results_2")     # Thư mục lưu bảng kết quả dữ liệu CSV

INPUT_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

# Các mức băng thông thử nghiệm của Encodec (kbps)
BANDWIDTHS = [1.5, 3.0, 6.0, 12.0]

# =====================================================================
# 1. THUẬT TOÁN TÁCH NGUỒN (STFT Hard Masking - Sửa liền mạch dải tần)
# =====================================================================
def separate(audio, sr):
    # Sử dụng nperseg=1024 để phân tích đồ thị phổ
    f, t, Zxx = stft(audio, sr, nperseg=1024)
    mag = np.abs(Zxx)
    
    bass = np.zeros_like(mag)
    vocal = np.zeros_like(mag)
    high = np.zeros_like(mag)
    
    # Sửa ranh giới liền mạch: không để khoảng hở từ 200Hz - 300Hz
    for i, freq in enumerate(f):
        if freq < 200:
            bass[i] = mag[i]
        elif 200 <= freq <= 3000:
            vocal[i] = mag[i]
        else:
            high[i] = mag[i]
            
    total = bass + vocal + high + 1e-9
    
    # Tạo Soft Mask khôi phục lại tín hiệu dạng phức (Complex)
    Z_bass = Zxx * (bass / total)
    Z_vocal = Zxx * (vocal / total)
    Z_high = Zxx * (high / total)
    
    _, bass_audio = istft(Z_bass, sr)
    _, vocal_audio = istft(Z_vocal, sr)
    _, high_audio = istft(Z_high, sr)
    
    return bass_audio, vocal_audio, high_audio

# =====================================================================
# 2. QUY TRÌNH NÉN VÀ GIẢI NÉN QUA ENCODEC (Cho 1 Stem)
# =====================================================================
def encodec_process(audio_np, sr, model, name, OUT_DIR):
    audio_np = audio_np.astype(np.float32)
    # Tạo tensor dạng [Channels, Samples] -> [1, T] cho dữ liệu Mono
    wav = torch.tensor(audio_np).unsqueeze(0) 
    
    # Ép chuỗi mẫu về Sample Rate của mô hình Encodec (24kHz)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    # Thêm chiều Batch thành cấu trúc [Batch, Channels, Samples] -> [1, 1, T]
    wav = wav.unsqueeze(0) 
    
    with torch.no_grad():
        encoded_frames = model.encode(wav)
        # Lưu file lượng tử hóa .pt xuống ổ đĩa
        encoded_path = OUT_DIR / f"{name}_encoded.pt"
        torch.save(encoded_frames, encoded_path)
        
    with torch.no_grad():
        decoded_audio = model.decode(encoded_frames)
        
    # Giải nén đưa về mảng phẳng numpy 1D (Mặc định ở 24kHz)
    rec = decoded_audio.squeeze().cpu().numpy()
    rec = rec.astype(np.float32)
    
    # Khử lỗi NaN / Vô cực nếu có
    rec = np.nan_to_num(rec)
    
    # Chuẩn hóa biên độ cục bộ cho từng stem
    max_val = np.max(np.abs(rec))
    if max_val > 0:
        rec = rec / max_val
        
    # Lưu file âm thanh đơn lẻ của từng stem ở tần số 24kHz
    wav_path = OUT_DIR / f"{name}_compressed.wav"
    sf.write(wav_path, rec, model.sample_rate)
    
    return rec, encoded_path

def calc_bitrate(file_path, duration):
    # Trả về dung lượng thực tế bao gồm cả metadata của PyTorch Tensor (.pt)
    size_bits = file_path.stat().st_size * 8
    return size_bits / duration / 1000  # kbps

# =====================================================================
# 3. PIPELINE ĐIỀU PHỐI CHÍNH
# =====================================================================
def main():
    # Quét toàn bộ file nhạc hợp lệ trong thư mục đầu vào
    audio_files = list(INPUT_DIR.glob("*.mp3")) + list(INPUT_DIR.glob("*.wav"))
    
    if not audio_files:
        print(f"[-] Thư mục '{INPUT_DIR}' trống! Hãy bỏ file nhạc test vào đây.")
        return

    # Khởi tạo mô hình Encodec mã hóa cấu trúc 24kHz
    model = EncodecModel.encodec_model_24khz()
    model.eval()
    
    for file_path in audio_files:
        print(f"\n=============================================")
        print(f"[+] ĐANG TIẾN HÀNH XỬ LÝ FILE: {file_path.name}")
        print(f"=============================================")
        
        audio, sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1) # Trộn kênh Stereo về Mono
            
        duration_sec = len(audio) / sr
        original_bits = len(audio) * 16  # Chuẩn PCM 16-bit Mono
        
        # Bước 1: Phân tách nguồn bằng thuật toán STFT (Chạy ở Sample Rate gốc)
        bass, vocal, high = separate(audio, sr)
        
        # Bước 2: Chuẩn bị file đối chứng chuẩn (Reference Audio) ở 24kHz
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        audio_24k = convert_audio(audio_tensor, sr, model.sample_rate, model.channels)
        ref_audio = audio_24k.squeeze().numpy()
        ref_audio = np.nan_to_num(ref_audio)
        ref_audio = ref_audio / (np.max(np.abs(ref_audio)) + 1e-9)
        
        rows = []
        
        # Bước 3: Vòng lặp quét qua từng mức Target Bandwidth
        for bw in BANDWIDTHS:
            model.set_target_bandwidth(bw)
            
            # Tiến hành nén/giải nén độc lập từng stem dựa trên mức băng thông
            stem_prefix = f"{file_path.stem}_{bw}"
            bass_rec, bass_file = encodec_process(bass, sr, model, f"bass_{stem_prefix}", OUT_DIR)
            vocal_rec, vocal_file = encodec_process(vocal, sr, model, f"vocal_{stem_prefix}", OUT_DIR)
            high_rec, high_file = encodec_process(high, sr, model, f"high_{stem_prefix}", OUT_DIR)
            
            # Đọc dung lượng bitrate file .pt trên ổ cứng
            bass_br = calc_bitrate(bass_file, duration_sec)
            vocal_br = calc_bitrate(vocal_file, duration_sec)
            high_br = calc_bitrate(high_file, duration_sec)
            
            # Đồng bộ ma trận độ dài giữa các dải sau nén
            min_len_stems = min(len(bass_rec), len(vocal_rec), len(high_rec))
            
            # Thực hiện cộng gộp các dải để tạo ra chuỗi tái cấu trúc (reconstructed)
            reconstructed = bass_rec[:min_len_stems] + vocal_rec[:min_len_stems] + high_rec[:min_len_stems]
            reconstructed = np.nan_to_num(reconstructed)
            
            # Khử peak biên độ, giới hạn chặt chẽ trong miền [-1.0, 1.0] chống vỡ tiếng
            if np.max(np.abs(reconstructed)) > 0:
                reconstructed = reconstructed / np.max(np.abs(reconstructed))
            reconstructed = np.clip(reconstructed, -1.0, 1.0)
            
            # SỬA LỖI ĐẦU RA KHÔNG CHẠY ĐƯỢC: Chuyển đổi ngược kết quả tái tạo từ 24kHz về 44.1kHz chuẩn hóa
            rec_tensor = torch.tensor(reconstructed, dtype=torch.float32).unsqueeze(0)
            rec_44k = convert_audio(rec_tensor, model.sample_rate, 44100, model.channels)
            reconstructed_44k = rec_44k.squeeze().numpy()
            reconstructed_44k = np.clip(reconstructed_44k, -1.0, 1.0) # Bảo vệ biên độ loa một lần nữa
            
            # Ghi file tổng hợp cuối cùng ở tần số 44.1kHz chuẩn để phát được ở mọi phần mềm nghe nhạc
            out_path = OUT_DIR / f"reconstructed_{file_path.stem}_{bw:.1f}kbps.wav"
            sf.write(out_path, reconstructed_44k, 44100)
            
            # Bước 4: Đồng bộ hóa độ dài với mảng Reference 24kHz để đo lường
            min_len_final = min(len(ref_audio), len(reconstructed))
            ref_aligned = ref_audio[:min_len_final]
            rec_aligned = reconstructed[:min_len_final]
            
            # Tính toán sai số toán học
            m = mse(ref_aligned, rec_aligned)
            s = snr_db(ref_aligned, rec_aligned)
            
            # Tổng băng thông lý thuyết tiêu tốn khi nén độc lập 3 file cùng lúc
            total_compressed_bandwidth_kbps = bw * 3 
            compressed_bits = (total_compressed_bandwidth_kbps * 1000) * duration_sec
            cr = compression_ratio(original_bits, compressed_bits)
            
            rows.append({
                "bandwidth_kbps": bw,
                "mse": m,
                "snr_db": s,
                "compression_ratio": cr,
                "bass_bitrate_file_pt": bass_br,
                "vocal_bitrate_file_pt": vocal_br,
                "high_bitrate_file_pt": high_br
            })
            print(f"[->] Hoàn thành mức cấu hình: {bw} kbps")
            
        # Xuất và ghi dữ liệu bảng thống kê kết quả cho bài nhạc hiện tại
        df = pd.DataFrame(rows)
        csv_path = TABLE_DIR / f"results_{file_path.stem}.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\n===== KẾT QUẢ PHÂN TÍCH: {file_path.stem} =====")
        print(df.to_string())
        print(f"[+] Bảng dữ liệu CSV đã được lưu tại: {csv_path}\n")
if __name__ == "__main__":
    main()
