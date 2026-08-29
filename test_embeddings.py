import torch
from MyCode.utils.image_transforms import DelayEmbedder, PatchEmbedder, STFTEmbedder, MRTIEmbedder

device = "cuda" if torch.cuda.is_available() else "cpu"
seq_len = 32
batch_size = 4
channels = 6

# Create sample time series
x = torch.randn(batch_size, seq_len, channels).to(device)

print(f"Input shape: {x.shape}")
print(f"Device: {device}\n")

# Test Delay Embedding
print("=== DELAY EMBEDDING ===")
embedder = DelayEmbedder(device, seq_len, delay=4, embedding=8)
img = embedder.ts_to_img(x)
reconstructed = embedder.img_to_ts(img)
print(f"Image shape: {img.shape}")
print(f"Reconstructed shape: {reconstructed.shape}\n")

# Test Patch Embedding
print("=== PATCH EMBEDDING ===")
embedder = PatchEmbedder(device, seq_len, patch_size=4, img_size=8)
img = embedder.ts_to_img(x)
reconstructed = embedder.img_to_ts(img)
print(f"Image shape: {img.shape}")
print(f"Reconstructed shape: {reconstructed.shape}\n")

# Test STFT Embedding
print("=== STFT EMBEDDING ===")
embedder = STFTEmbedder(device, seq_len, n_fft=16, hop_length=4)
# STFT needs min/max caching like ImagenTime
embedder.cache_min_max_params(x)
img = embedder.ts_to_img(x)
reconstructed = embedder.img_to_ts(img)
print(f"Image shape: {img.shape}")
print(f"Reconstructed shape: {reconstructed.shape}\n")

# Test MRTI Embedding
print("=== MRTI EMBEDDING ===")
embedder = MRTIEmbedder(device, seq_len, num_scales=3, num_periods=3)
img = embedder.ts_to_img(x)
reconstructed = embedder.img_to_ts(img)
print(f"Image shape: {img.shape}")
print(f"Reconstructed shape: {reconstructed.shape}\n")

print("✓ All embeddings working!")
