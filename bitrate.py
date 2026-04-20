import pandas as pd
import matplotlib.pyplot as plt

# đọc dữ liệu
df = pd.read_csv("results_hard/stem_comparison.csv")

# tạo 3 subplot (mỗi stem 1 đồ thị)
fig, axs = plt.subplots(3, 1)

# ===== Bass =====
axs[0].plot(df["bandwidth_kbps"], df["bass_bitrate"], marker='o')
axs[0].set_title("Bass Bitrate vs Bandwidth")
axs[0].set_xlabel("Bandwidth (kbps)")
axs[0].set_ylabel("Bitrate (kbps)")
axs[0].grid()

# ===== Vocal =====
axs[1].plot(df["bandwidth_kbps"], df["vocal_bitrate"], marker='s')
axs[1].set_title("Vocal Bitrate vs Bandwidth")
axs[1].set_xlabel("Bandwidth (kbps)")
axs[1].set_ylabel("Bitrate (kbps)")
axs[1].grid()

# ===== High =====
axs[2].plot(df["bandwidth_kbps"], df["high_bitrate"], marker='^')
axs[2].set_title("High Bitrate vs Bandwidth")
axs[2].set_xlabel("Bandwidth (kbps)")
axs[2].set_ylabel("Bitrate (kbps)")
axs[2].grid()

plt.tight_layout()
plt.show()