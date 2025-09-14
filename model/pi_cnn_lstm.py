import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride=2, padding=2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

def physics_aware_activation(x: torch.Tensor, lower_bound: float, upper_bound: float) -> torch.Tensor:
    return lower_bound + (upper_bound - lower_bound) * (torch.tanh(x) + 1) / 2

class CnnFeatureExtractor(nn.Module):
    def __init__(self,
                 input_channels: int = 1,
                 conv_channels: int = 16,
                 num_conv_layers: int = 4,
                 kernel_size: int = 5):
        super().__init__()
        self.output_channels = conv_channels * (2 ** (num_conv_layers - 1))
        
        # Build conv layers
        conv_layers = [BasicConvBlock(input_channels, conv_channels, kernel_size)]
        for i in range(num_conv_layers - 1):
            conv_layers.append(BasicConvBlock(
                conv_channels * (2 ** i), 
                conv_channels * (2 ** (i + 1)), 
                kernel_size
            ))
        conv_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)

class LstmSequenceProcessor(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 bidirectional: bool = False,
                 use_layer_norm: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_layer_norm = use_layer_norm
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Optional layer norm
        self.layer_norm = (nn.LayerNorm(hidden_size * self.num_directions) 
                          if use_layer_norm else None)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Initialize states
        hidden_state = torch.zeros(
            self.num_layers * self.num_directions, 
            batch_size, 
            self.hidden_size
        ).to(x.device)
        cell_state = torch.zeros(
            self.num_layers * self.num_directions, 
            batch_size, 
            self.hidden_size
        ).to(x.device)
        
        # LSTM processing
        lstm_output, _ = self.lstm(x, (hidden_state, cell_state))
        
        # Take last time step
        x = lstm_output[:, -1, :]
        
        # Optional layer norm
        if self.layer_norm is not None:
            x = self.layer_norm(x)
            
        return x

class PhysicsInformedMlp(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 output_dim: int = 5):
        super().__init__()
        self.output_dim = output_dim
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, output_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process through layers
        x = self.classifier(x)
        
        # Apply physics constraints
        physics_constrained_output = torch.zeros_like(x)
        physics_constrained_output[:, 0] = physics_aware_activation(x[:, 0], 0.0, 0.5)
        physics_constrained_output[:, 1] = (167.0 / 2700 / 900) * physics_aware_activation(x[:, 1], 0.3, 1.0)
        physics_constrained_output[:, 2] = 167.0 * physics_aware_activation(x[:, 2], 0.826, 1.0)
        physics_constrained_output[:, 3] = 1e-4 * physics_aware_activation(x[:, 3], 0.5, 2)
        physics_constrained_output[:, 4] = physics_aware_activation(x[:, 4], 0.0, 0.5)
        
        return physics_constrained_output

class CnnLstmPinnModel(nn.Module):
    def __init__(self,
                 input_channels: int = 1,
                 conv_channels: int = 16,
                 num_conv_layers: int = 4,
                 lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2,
                 kernel_size: int = 5,
                 output_dim: int = 5,
                 use_layer_norm: bool = False,
                 bidirectional: bool = False):
        super().__init__()
        
        # CNN Feature Extractor
        self.spatial_encoder = CnnFeatureExtractor(
            input_channels=input_channels,
            conv_channels=conv_channels,
            num_conv_layers=num_conv_layers,
            kernel_size=kernel_size
        )
        
        # LSTM Sequence Processor
        self.temporal_encoder = LstmSequenceProcessor(
            input_size=self.spatial_encoder.output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=bidirectional,
            use_layer_norm=use_layer_norm
        )
        
        # Physics-Informed MLP
        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)
        self.decoder = PhysicsInformedMlp(
            input_size=lstm_output_size,
            hidden_size=lstm_hidden_size // 2,
            output_dim=output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract input dimensions
        batch_size, seq_len, channels, height, width = x.shape
        
        # Spatial encoding
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.spatial_encoder(x)
        x = x.view(batch_size, seq_len, -1)
        
        # Temporal encoding
        x = self.temporal_encoder(x)
        
        # Physics-informed decoding
        x = self.decoder(x)
        
        return x