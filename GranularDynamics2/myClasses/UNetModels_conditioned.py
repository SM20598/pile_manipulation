import torch.nn as nn
import torch
from .UNetModels import ConvBlock

############################################
# A SIMPLTE PHYSICS CONDITIONED UNET MODEL #
############################################

class UNetConditioned(nn.Module):
    
    def __init__(
        self,
        in_channels=2, # state(i) + action(i) layers
        physics_dim=6, # scene physics
        out_channels=1 # state(i+1) layer
    ):
        super().__init__()
        
        total_input_channels = in_channels + physics_dim
        
        # Encoder
        self.enc1 = ConvBlock(in_channels=total_input_channels, out_channels=64)
        self.enc2 = ConvBlock(in_channels=64, out_channels=128)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Bottleneck
        self.bottle = ConvBlock(in_channels=128, out_channels=256)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(256, 128)
        
        self.up2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        
    def forward(self, x, physics):
        B, _, H, W = x.size()
        physics = physics[:, :, None, None]
        physics = physics.expand(B, physics.size(1), H, W)
        
        x = torch.cat([x, physics], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1) # down
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2) # down
        
        # Bottleneck
        b = self.bottle(p2) # middle
        
        # Decoder
        u1 = self.up1(b) # up
        c1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(c1)
        
        u2 = self.up2(d1)
        c2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(c2)
        
        return self.out(d2)
        
###############################
# FiLM CONDITIONAL UNET MODEL #
###############################

class FiLMGenerator(nn.Module):
    def __init__(
            self,
            physics_dim,
            hidden=128,
            film_dims=(64, 128, 256
    )):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(physics_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        
        self.to_film = nn.ModuleDict({
            "enc1": nn.Linear(hidden, film_dims[0] * 2),
            "enc2": nn.Linear(hidden, film_dims[1] * 2),
            "bottle": nn.Linear(hidden, film_dims[2] * 2),
            "dec1": nn.Linear(hidden, film_dims[1] * 2),
            "dec2": nn.Linear(hidden, film_dims[0] * 2),
        })
        
    def forward(self, physics):
        h = self.mlp(physics)
        
        film = {}
        
        for key, layer in self.to_film.items():
            gb = layer(h)
            gamma, beta = gb.chunk(2, dim=-1)
            film[key] = (gamma, beta)
        return film
    
class UNetFiLM(nn.Module):
    def __init__(
        self,
        in_channels=2,
        physics_dim=3,
        out_channels=1,
    ):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.enc2 = ConvBlock(in_channels=64, out_channels=128)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Bottleneck
        self.bottle = ConvBlock(in_channels=128, out_channels=256)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(256, 128)
        
        self.up2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        
        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        
        self.film = FiLMGenerator(physics_dim=physics_dim, film_dims=(64, 128, 256))
    
    def apply_film(self, x, gamma, beta):
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * gamma + beta
    
    def forward(self, x, physics):
        B, _, H, W = x.size()
        
        film_params = self.film(physics)
        
        # Encoder
        e1 = self.apply_film(self.enc1(x), *film_params['enc1'])
        p1 = self.pool(e1) # down
        
        e2 = self.apply_film(self.enc2(p1), *film_params['enc2'])
        p2 = self.pool(e2) # down
        
        # Bottleneck
        b = self.bottle(p2) # middle
        b = self.apply_film(b, *film_params['bottle'])
        
        # Decoder
        u1 = self.up1(b) # up
        c1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(c1)
        d1 = self.apply_film(d1, *film_params['dec1'])
        
        u2 = self.up2(d1)
        c2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(c2)
        d2 = self.apply_film(d2, *film_params['dec2'])
        
        return self.out(d2)