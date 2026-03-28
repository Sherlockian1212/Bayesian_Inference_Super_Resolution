import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTBlock(nn.Module):
    def __init__(self, dim, depth=4, heads=4, mlp_dim=128, patch_size=16, attn_drop=0.0, ff_drop=0.1, max_patches=1024):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim

        # Patchify + Unpatchify
        self.to_patch = nn.Conv2d(dim, dim, patch_size, stride=patch_size)
        self.to_img = nn.ConvTranspose2d(dim, dim, patch_size, stride=patch_size)

        # Positional embedding (init with max_patches, resize later if needed)
        self.max_patches = max_patches
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_patches, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=ff_drop,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x, is_transpose=True):
        B, C, H, W = x.shape
        x = self.to_patch(x)                    # [B,C,Hp,Wp]
        Hp, Wp = x.shape[2], x.shape[3]
        N = Hp * Wp

        x = x.flatten(2).transpose(1, 2)        # [B, N, C]

        # Resize pos_embedding nếu N khác self.max_patches
        if N != self.pos_embedding.shape[1]:
            pos_emb = self.pos_embedding
            # interpolate to new size
            pos_emb = pos_emb.transpose(1, 2).view(1, self.dim, int(self.max_patches**0.5), int(self.max_patches**0.5))
            pos_emb = F.interpolate(pos_emb, size=(Hp, Wp), mode='bilinear', align_corners=False)
            pos_emb = pos_emb.flatten(2).transpose(1, 2)  # [1, N, C]
        else:
            pos_emb = self.pos_embedding[:, :N, :]

        # add position
        x = x + pos_emb

        # Transformer stack
        x = self.transformer(x)                 # [B, N, C]

        # reshape image
        if is_transpose:
            x = x.transpose(1, 2).view(B, C, Hp, Wp)
            x = self.to_img(x)                      # [B,C,H,W]
        return x
