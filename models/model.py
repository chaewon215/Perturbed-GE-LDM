import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Union, Tuple
from transformers import AutoModel, AutoTokenizer

import numpy as np
import pandas as pd
import os


class MoleculeModel(nn.Module):
    def __init__(self, args, context_dim=576, ge_dim=978, hidden_dim=512, timesteps=1000):
        super().__init__()
        self.n_timesteps = timesteps        # 몇 번의 diffusion step을 수행할지 (noise를 몇 번 추가할지)
        self.device = args.device
        self.mode = args.mode
        
        self.latent_dim = 256
        
        self.ge_dim = ge_dim  # Gene expression dimension
        
        self.normalize_basal_gex_mean()   # basal gene expression data를 normalize

        self.register_buffer("betas", torch.linspace(1e-5, 0.01, self.n_timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("alpha_bars_prev", torch.cat([torch.ones(1), self.alpha_bars[:-1]]))


        # posterior 계산에 필요한 계수들
        self.register_buffer(
            "posterior_variance",
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        # log variance (clip은 여기선 생략, 나중에 필요시 추가 가능)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]))
        )
        self.register_buffer(
            "posterior_mean_coef1",
            self.betas * torch.sqrt(self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - self.alpha_bars_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bars)
        )

        # define the molformer model
        self.molformer = AutoModel.from_pretrained(
            "ibm/MoLFormer-XL-both-10pct",
            trust_remote_code=True,
            deterministic_eval=True
        ).to(self.device).eval()
        for param in self.molformer.parameters():
            param.requires_grad = False

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

        # self.omics_conv1 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)

        self.mol_ffn = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
        )
        
        self.basal_ffn = nn.Sequential(
            nn.Linear(978, 512),
            nn.BatchNorm1d(512),
            # nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            # nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            # nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 256),
        )

        self.dose_ffn = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )
        self.time_ffn = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 512),
            nn.BatchNorm1d(512),
            # nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            # nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
        )

        self.denoiser_net = Denoiser(latent_dim=self.latent_dim, hidden_dim=hidden_dim, context_dim=256)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        

    def normalize_basal_gex_mean(self):
        """
        Get the mean and standard deviation of the control gene expression data.

        :param ctrl_gex: The control gene expression data.
        :return: The mean and standard deviation of the control gene expression data.
        """
        self.ctrl_gex_df = pd.read_csv(
            "../Data/LINCS_L1000/Data_PRnet/ctrl_gex_norm_log.csv", index_col=0
        )
        self.ctrl_gex_std, self.ctrl_gex_mean = torch.std_mean(torch.tensor(self.ctrl_gex_df.values, dtype=torch.float32), dim=0, unbiased=False)
        self.ctrl_gex_std = self.ctrl_gex_std.to(self.device)
        self.ctrl_gex_mean = self.ctrl_gex_mean.to(self.device)
        
        # Normalize the control gene expression data
        self.ctrl_gex = torch.tensor(self.ctrl_gex_df.values, dtype=torch.float32)
        self.ctrl_gex = self.ctrl_gex.to(self.device)
        self.ctrl_gex = (self.ctrl_gex - self.ctrl_gex_mean) / self.ctrl_gex_std
        self.cell_ids = list(self.ctrl_gex_df.index)  # e.g., ['A549', 'MCF7', ...]
        
        self.cell2idx = {cell_id: i for i, cell_id in enumerate(self.cell_ids)}


    def encode_context(self, basal_ge, smiles, dose, time):
        
        # Compound-specific features
        inputs = self.tokenizer(smiles, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            mol_embedding = self.molformer(**inputs).pooler_output  # (B, 768)
        mol_embedding = self.mol_ffn(mol_embedding)  # (B, 512)
        
        # Cell-specific features        
        basal_ge = self.basal_ffn(basal_ge)  # (B, 512)

        # Condition features
        dose = self.dose_ffn(dose.to(self.device))  # (B, 64)
        time = self.time_ffn(time.to(self.device))  # (B, 64)

        # Combine all features to form context vector
        context = torch.cat([mol_embedding, basal_ge, time, dose, ], dim=1)  # (B, 512 + 512 + 64 + 64 ) = (B, 1234)
        
        context = context.to(self.device)
        return self.context_encoder(context)  # (B, hidden_dim)


    def forward_posterior(self, z_0, z_t, t):
        """
        Compute q(z_{t-1} | z_t, z_0) = N(posterior_mean, posterior_variance)
        """
        # shape: (B, 1)
        coef1 = self.posterior_mean_coef1[t].view(-1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1)
        posterior_mean = coef1 * z_0 + coef2 * z_t

        posterior_log_var = self.posterior_log_variance_clipped[t].view(-1, 1)

        return posterior_mean, posterior_log_var
        
        
    def forward_diffusion(self, z_0, t):
        """
        (For mu, variance prediction model)
        Latent space z_0 to noisy latent space z_t.
        """
        noise = torch.randn_like(z_0)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1).to(self.device)
        z_t = torch.sqrt(alpha_bar_t) * z_0 + torch.sqrt(1 - alpha_bar_t) * noise
        return z_t, noise


    def reverse_diffusion(self, x, basal_ge, smiles, dose, time, t, save, count):
        
        if self.mode == "pred_mu_var":
            mu, log_var = self.denoiser(x, basal_ge, smiles, dose, time, t, save, count)
        elif self.mode == "pred_mu_v":
            mu, log_var, v, log_beta_t, log_beta_tilde = self.denoiser_pred_v(x, basal_ge, smiles, dose, time, t, save, count)
            
        sigma = torch.exp(0.5 * log_var)

        # sampling with noise (stochastic) or omit noise (DDIM-like deterministic)
        eps = torch.randn_like(mu)
        x_prev = mu + sigma * eps

        if self.mode == "pred_mu_var":
            return x_prev, mu, sigma, 
        elif self.mode == "pred_mu_v":
            return x_prev, mu, sigma, v, log_beta_t, log_beta_tilde

    
    def denoiser(self, noisy_x, basal_ge, smiles, dose, time, t, save=False, count=0):
        context = self.encode_context(basal_ge, smiles, dose, time)
        
        t_emb = self.time_mlp(t.float().view(-1, 1))
        
        out = self.denoiser_net(noisy_x, context, t_emb)  # shape: (B, latent_dim * 2)
        mu, log_var = torch.chunk(out, 2, dim=1)  # (B, latent_dim), (B, latent_dim)

        # Optional: clip log_var to prevent extreme values
        log_var = torch.clamp(log_var, min=-10, max=10)

        return mu, log_var
    

    def denoiser_pred_v(self, noisy_x, basal_ge, smiles, dose, time, t, save=False, count=0):
        context = self.encode_context(basal_ge, smiles, dose, time)
        t_emb = self.time_mlp(t.float().view(-1, 1))
        
        out = self.denoiser_net(noisy_x, context, t_emb)
        mu, v = torch.chunk(out, 2, dim=1)
        v = torch.sigmoid(v)

        beta_t = self.betas[t].view(-1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1)
        alpha_bar_prev = self.alpha_bars[torch.clamp(t - 1, 0)].view(-1, 1)
        beta_t_tilde = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t

        eps = 1e-20
        log_beta_t = torch.log(beta_t.clamp(min=eps))
        log_beta_tilde  = torch.log(beta_t_tilde.clamp(min=eps))

        log_var = v * log_beta_t + (1 - v) * log_beta_tilde

        return mu, log_var, v, log_beta_t, log_beta_tilde
    
    
    def forward_diffusion_np(self, x, t):
        """
        (For noise prediction model)
        Latent space x_0 to noisy latent space x_t.       
        """
        noise = torch.randn_like(x).to(self.device)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1).to(self.device)
        noisy_x = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

        return noisy_x, noise
    

    def reverse_diffusion_np(self, x, basal_ge, smiles, dose, time, t):
        """
        (For noise prediction model)
        Predict the noise that added to x_{t-1} using x_t.
        Return the reconstructed x_{t-1}.
        """
        predicted_noise = self.denoiser(x, basal_ge, smiles, dose, time, t)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1).to(self.device)
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

        t_prev = torch.clamp(t - 1, min=0)
        alpha_bar_prev = self.alpha_bars[t_prev].view(-1, 1).to(self.device)

        return torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * predicted_noise
    

    def sample(self, cell, basal_ge, smiles, dose, time, batch_size, n_steps=None, save=False, count=0):
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        init_z = z.clone()  # Save initial z for potential use later
        steps = list(torch.linspace(0, self.n_timesteps - 1, n_steps or 50, dtype=torch.int64))

        for t in reversed(steps):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            if self.mode == "pred_mu_var":
                z, mu, sigma = self.reverse_diffusion(z, basal_ge, smiles, dose, time, t_batch, save=save, count=count)
            elif self.mode == "pred_mu_v":
                z, mu, sigma, v, log_beta_t, log_beta_tilde = self.reverse_diffusion(z, basal_ge, smiles, dose, time, t_batch, save=save, count=count)

            if save:
                info_df = pd.DataFrame({
                    'cell': [c[0] for c in cell],
                    'smiles': smiles,
                    'dose': dose.cpu().numpy().reshape(-1),
                    'time': time.cpu().numpy().reshape(-1),
                })
                info_df.to_csv(f'./fold_1/{count}/info.csv', index=False)
                
                torch.save(init_z, f'./fold_1/{count}/init_z.pt')
                if (steps.index(t) % 10 == 9) or steps.index(t) == 0:
                    print(f"Step {steps.index(t)}: t={t}, x={z.mean().item()}")
                    if not os.path.exists(f'./fold_1/{count}'):
                        os.makedirs(f'./fold_1/{count}')
                    torch.save(z, f'./fold_1/{count}/step_{steps.index(t)}.pt')

        if self.mode == "pred_mu_var":
            return z, mu, sigma,
        elif self.mode == "pred_mu_v":
            return z, mu, sigma, v, log_beta_t, log_beta_tilde



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.GroupNorm(8, in_channels),
#             nn.SiLU(),
#             nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.GroupNorm(8, out_channels),
#             nn.SiLU(),
#             nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
#         )
#         self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

#     def forward(self, x):
#         return self.block(x) + self.residual(x)

# class CrossAttention(nn.Module):
#     def __init__(self, dim, context_dim, heads=4):
#         super().__init__()
#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(context_dim, dim)
#         self.v_proj = nn.Linear(context_dim, dim)
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
#         self.norm_x = nn.LayerNorm(dim)
#         self.norm_context = nn.LayerNorm(context_dim)

#     def forward(self, x, context):
#         # x: (B, N, D), context: (B, C, D_context)
#         x = self.norm_x(x)
#         context = self.norm_context(context)
#         q = self.q_proj(x)
#         k = self.k_proj(context)
#         v = self.v_proj(context)
#         out, _ = self.attn(q, k, v)
#         return out + x

# class GeneDenoiserUNet(nn.Module):
#     def __init__(self, latent_dim=256, ge_dim=978, comp_dim=768, noise_dim=64, t_dim=128, base_dim=128):
#         super().__init__()

#         # project all inputs to base_dim
#         self.input_proj = nn.Linear(latent_dim, base_dim)
#         self.t_proj = nn.L inear(t_dim, base_dim)
#         self.n_proj = nn.Linear(noise_dim, base_dim)
#         self.omics_proj = nn.Linear(ge_dim, base_dim)
#         self.comp_proj = nn.Linear(comp_dim, base_dim)

#         # U-Net encoder
#         self.down1 = ResBlock(base_dim, base_dim * 2)
#         self.down2 = ResBlock(base_dim * 2, base_dim * 4)

#         # middle block + cross attention
#         self.middle = ResBlock(base_dim * 4, base_dim * 4)
#         self.cross_attn = CrossAttention(base_dim * 4, base_dim)

#         # U-Net decoder
#         self.up1 = ResBlock(base_dim * 4, base_dim * 2)
#         self.up2 = ResBlock(base_dim * 2, base_dim)
#         self.final = nn.Conv1d(base_dim, latent_dim, kernel_size=1)

#     def forward(self, x_t, omics, mol_embedding, t_emb, rand_noise):
#         """
#         Denoiser U-Net forward pass.
#         :param x_t: Noisy latent space input (B, latent_dim).
#         :param omics: Omics data (B, 3, 978).
#         :param mol_embedding: Compound embedding (B, 768).
#         :param t_emb: Time embedding (B, 128).
#         :param rand_noise: Random noise (B, 64).
#         :return: Denoised latent space output (B, latent_dim).
#         """
        
#         B = x_t.size(0)

#         # Embed inputs
#         x = self.input_proj(x_t).unsqueeze(-1)  # (B, D, 1)
#         t = self.t_proj(t_emb).unsqueeze(-1)
#         n = self.n_proj(rand_noise).unsqueeze(-1)
#         x = x + t + n

#         # context sequence: omics (B, 3, 978) and compound (B, 768)
#         omics = omics.view(B * 3, -1)  # (B*3, 978)
#         omics = self.omics_proj(omics).view(B, 3, -1)  # (B, 3, D)
#         compound = self.comp_proj(mol_embedding).unsqueeze(1)  # (B, 1, D)
#         context = torch.cat([omics, compound], dim=1)  # (B, 4, D)

#         # U-Net downsampling
#         d1 = self.down1(x)  # (B, 2D, 1)
#         d2 = self.down2(d1)  # (B, 4D, 1)

#         # middle block
#         m = self.middle(d2)  # (B, 4D, 1)
#         m_flat = m.transpose(1, 2)  # (B, 1, 4D)
#         attn_out = self.cross_attn(m_flat, context)  # (B, 1, 4D)
#         m = attn_out.transpose(1, 2)  # (B, 4D, 1)

#         # U-Net upsampling
#         u1 = self.up1(m + d2)  # skip connection
#         u2 = self.up2(u1 + d1)
#         out = self.final(u2).squeeze(-1)  # (B, latent_dim)

#         return out



import torch
import torch.nn as nn
import torch.nn.functional as F


class Denoiser(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, context_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(latent_dim + context_dim + 128, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        # self.fc1_bn = nn.LayerNorm(hidden_dim)
        self.fc1_lrelu = nn.LeakyReLU()
        self.fc1_dropout = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(hidden_dim + context_dim, hidden_dim)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim)
        # self.fc2_bn = nn.LayerNorm(hidden_dim)
        self.fc2_lrelu = nn.LeakyReLU()
        self.fc2_dropout = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(hidden_dim + context_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_bn = nn.BatchNorm1d(hidden_dim)
        # self.fc3_bn = nn.LayerNorm(hidden_dim)
        self.fc3_lrelu = nn.LeakyReLU()
        self.fc3_dropout = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(hidden_dim, latent_dim * 2)
    
    def forward(self, noisy_x, context, t_emb):
        
        input_tensor = torch.cat([noisy_x, context, t_emb], dim=1)  # Concatenate inputs
        
        out = self.fc1(input_tensor)
        out = self.fc1_bn(out)
        out = self.fc1_lrelu(out)
        out = self.fc1_dropout(out)
        
        out = torch.cat([out, context], dim=1)  # 512 + 256 = 768
        out = self.fc2(out)
        out = self.fc2_bn(out)
        out = self.fc2_lrelu(out)
        out = self.fc2_dropout(out)
        
        out = torch.cat([out, context], dim=1)  # 512 + 256 = 768
        out = self.fc3(out)
        out = self.fc3_bn(out)
        out = self.fc3_lrelu(out)
        out = self.fc3_dropout(out)
        
        out = self.fc4(out)  # Final output layer
        
        return out  # Return mean and log variance

