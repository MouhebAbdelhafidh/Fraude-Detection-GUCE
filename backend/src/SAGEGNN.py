import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. Hyperparameters & Settings
# ==========================================
DATASET_PATH = "./data/cat"
IMAGE_SIZE = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 20
TIMESTEPS = 300  # Number of diffusion time steps

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. Diffusion Beta Schedule (Linear)
# ==========================================
betas = torch.linspace(1e-4, 0.02, TIMESTEPS).to(DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def q_sample(x_0, t, noise=None):
    """Forward Process: Adds noise to clean image x_0 at time step t"""
    if noise is None:
        noise = torch.randn_like(x_0)
    
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    noisy_x = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
    return noisy_x, noise


# ==========================================
# 3. Sinusoidal Time Embedding
# ==========================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=DEVICE) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ==========================================
# 4. Conditional U-Net Architecture (Img2Img)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.relu(self.conv1(x))
        # Inject time embedding into feature maps
        time_feats = self.relu(self.time_mlp(t_emb))[:, :, None, None]
        h = h + time_feats
        h = self.conv2(h)
        return self.relu(h + self.shortcut(x))


class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, time_dim=128):
        """
        in_channels = 6 because we concatenate:
        3 channels (noisy image x_t) + 3 channels (condition image x_cond)
        """
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Encoder
        self.inc = ResidualBlock(in_channels, 64, time_dim)
        self.down1 = nn.Conv2d(64, 64, 4, stride=2, padding=1)  # 64x64
        self.enc1 = ResidualBlock(64, 128, time_dim)
        
        self.down2 = nn.Conv2d(128, 128, 4, stride=2, padding=1) # 32x32
        self.enc2 = ResidualBlock(128, 256, time_dim)

        # Bottleneck
        self.bottleneck = ResidualBlock(256, 256, time_dim)

        # Decoder + Skip Connections
        self.up2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(256, 128, time_dim)

        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(128, 64, time_dim)

        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x_t, x_cond, t):
        t_emb = self.time_mlp(t)
        
        # Concatenate noisy image and conditioning image along channel axis
        x = torch.cat([x_t, x_cond], dim=1)  # (B, 6, H, W)

        # Encoder
        e1 = self.inc(x, t_emb)
        e2 = self.enc1(self.down1(e1), t_emb)
        b = self.bottleneck(self.enc2(self.down2(e2), t_emb), t_emb)

        # Decoder + Skip Connections
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, t_emb)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1, t_emb)

        return self.outc(d1)


# ==========================================
# 5. Sampling / Inference Procedure
# ==========================================
@torch.no_grad()
def sample_img2img(model, x_cond, timesteps=TIMESTEPS):
    """Generates a clean image starting from noise, guided by x_cond"""
    model.eval()
    batch_size = x_cond.shape[0]
    
    # Start from pure noise x_T
    x_t = torch.randn_like(x_cond).to(DEVICE)

    for t_idx in reversed(range(timesteps)):
        t = torch.full((batch_size,), t_idx, device=DEVICE, dtype=torch.long)
        
        # Predict noise
        predicted_noise = model(x_t, x_cond, t)
        
        # Reverse Diffusion Formula: x_{t-1} calculation
        beta_t = betas[t_idx]
        alpha_t = alphas[t_idx]
        alpha_cumprod_t = alphas_cumprod[t_idx]
        
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)
        
        if t_idx > 0:
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(beta_t) * noise
        else:
            x_t = mean

    return torch.clamp(x_t, -1.0, 1.0)


# ==========================================
# 6. Dataset Loader (Creates Synthetic Img2Img Pairs)
# ==========================================
class Img2ImgDataset(Dataset):
    """Loads images and creates a degraded version (Blur/Low-Res) as x_cond"""
    def __init__(self, folder_path, image_size=128):
        extensions = ("*.jpg", "*.jpeg", "*.png")
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

        self.clean_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        # Condition Transform: Downscales & Blurs image to create an Img2Img task
        self.cond_transform = transforms.Compose([
            transforms.Resize((image_size // 4, image_size // 4)), # Downscale
            transforms.Resize((image_size, image_size)),           # Upscale back (Pixelated)
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        raw_img = Image.open(img_path).convert("RGB")
        
        x_clean = self.clean_transform(raw_img)
        x_cond = self.cond_transform(raw_img)  # Corrupted/Low-Res conditioning input
        
        return x_clean, x_cond


# ==========================================
# 7. Training Pipeline
# ==========================================
if __name__ == "__main__":
    try:
        dataset = Img2ImgDataset(DATASET_PATH, IMAGE_SIZE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        print(f"Loaded {len(dataset)} images for Img2Img Diffusion.")

        model = ConditionalUNet().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        print(f"\n--- Training Conditional Diffusion Model ({EPOCHS} Epochs) ---")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0

            for x_clean, x_cond in dataloader:
                x_clean, x_cond = x_clean.to(DEVICE), x_cond.to(DEVICE)
                
                # Sample random time steps t
                t = torch.randint(0, TIMESTEPS, (x_clean.shape[0],), device=DEVICE).long()
                
                # Forward Process: Add noise to clean image x_clean
                x_noisy, true_noise = q_sample(x_clean, t)
                
                # Model predicts noise given (x_noisy, x_cond, t)
                predicted_noise = model(x_noisy, x_cond, t)
                
                loss = criterion(predicted_noise, true_noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch:02d}/{EPOCHS}] | MSE Loss: {avg_loss:.6f}")

        # --- Display Results ---
        print("\n--- Running Img2Img Inference ---")
        model.eval()
        x_clean_sample, x_cond_sample = next(iter(dataloader))
        x_cond_sample = x_cond_sample[:4].to(DEVICE)
        
        generated_clean = sample_img2img(model, x_cond_sample)

        # Plot Ground Truth, Condition, and Diffusion Result
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for i in range(4):
            # Denormalize from [-1, 1] to [0, 1]
            cond_img = (x_cond_sample[i].cpu().permute(1, 2, 0).numpy() + 1) / 2.0
            gen_img = (generated_clean[i].cpu().permute(1, 2, 0).numpy() + 1) / 2.0
            gt_img = (x_clean_sample[i].permute(1, 2, 0).numpy() + 1) / 2.0

            axes[0, i].imshow(cond_img)
            axes[0, i].axis("off")
            if i == 0: axes[0, i].set_title("Input Condition (x_cond)", fontweight='bold')

            axes[1, i].imshow(gen_img)
            axes[1, i].axis("off")
            if i == 0: axes[1, i].set_title("Diffusion Output (Restored)", fontweight='bold')

            axes[2, i].imshow(gt_img)
            axes[2, i].axis("off")
            if i == 0: axes[2, i].set_title("Ground Truth (Target)", fontweight='bold')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"❌ {e}")