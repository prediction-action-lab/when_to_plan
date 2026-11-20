import torch.nn as nn
import torch


class ModelFreeActorImage(nn.Module):
    def __init__(self, observation_space, features_dim):
        super(ModelFreeActorImage, self).__init__()

        # observation_space.shape is (H, W, C)
        n_input_channels = observation_space.shape[2]

        self.cnn_model = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        with torch.no_grad():
            # Test with (B, C, H, W)
            h, w = observation_space.shape[0], observation_space.shape[1]
            sample = torch.zeros(1, n_input_channels, h, w)
            n_flatten = self.cnn_model(sample).shape[1]

        self.mlp_model = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
        )

    def forward(self, observations):
        # Add batch dimension if there isn't one
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(0)

        x = self.cnn_model(observations)
        x = self.mlp_model(x)
        return x


class ModelFreeActorTensor(nn.Module):
    def __init__(self, observation_space, features_dim):
        super(ModelFreeActorTensor, self).__init__()

        # Calculate total input dimension by flattening the observation space
        input_dim = 1
        for dim in observation_space.shape:
            input_dim *= dim

        self.mlp_model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
        )

    def forward(self, observations):
        # Add batch dimension if there isn't one
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(0)

        x = self.mlp_model(observations)
        return x


class ModelFreeActorTensorContinuous(nn.Module):
    def __init__(self, observation_space, action_dim):
        super(ModelFreeActorTensorContinuous, self).__init__()

        # Calculate total input dimension by flattening the observation space
        input_dim = 1
        for dim in observation_space.shape:
            input_dim *= dim

        # Shared feature extractor
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Direct action prediction for behavior cloning
        self.action_layer = nn.Linear(512, action_dim)

    def forward(self, observations):
        # Add batch dimension if there isn't one
        if len(observations.shape) == 1:
            observations = observations.unsqueeze(0)

        # Flatten observations
        x = observations.flatten(start_dim=1)
        x = self.shared_mlp(x)
        action = self.action_layer(x)

        return action


class ModelFreeActorImageContinuous(nn.Module):
    """
    Image-based policy network for continuous action spaces.
    For behavior cloning, directly outputs action prediction (no distribution).

    Architecture notes
    ------------------
    Designed for *memorisation* of a small fixed task set (e.g. 10 tasks).
    Key choices:
    * No Dropout — dropout prevents per-task memorisation by randomly zeroing
      activations; it is actively harmful when the goal is to learn every
      training configuration perfectly.
    * No BatchNorm — BN introduces stochastic per-batch normalisation noise
      that disrupts tight convergence on individual configurations.
    * AdaptiveAvgPool2d(8) collapses spatial dims to 8×8, giving a
      256×64 = 16 384 feature vector — large enough to encode fine-grained
      per-task appearance differences without being wasteful.
    * Wide MLP (1024 → 512) provides the capacity needed to store distinct
      joint-angle trajectories for every cached task.
    * Tanh output constrains predictions to [-1, 1], matching the normalised
      joint-angle targets produced by get_plan(pretrain=True).
    """

    def __init__(self, observation_space, action_dim):
        super(ModelFreeActorImageContinuous, self).__init__()

        # observation_space.shape is (H, W, C) for HWC format (gymnasium standard)
        n_input_channels = observation_space.shape[2]

        self.cnn_model = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),  
            nn.Flatten(),
        )

        with torch.no_grad():
            h, w, c = observation_space.shape
            sample = torch.zeros(1, c, h, w)
            n_flatten = self.cnn_model(sample).shape[1]

        # ── High-capacity MLP head (no Dropout) ───────────────────────────
        self.shared_mlp = nn.Sequential(
            nn.Linear(n_flatten, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Tanh bounds output to [-1, 1] matching the normalised joint targets.
        self.action_layer = nn.Sequential(
            nn.Linear(512, action_dim),
            nn.Tanh(),
        )

    def forward(self, observations):
        # Add batch dimension if there isn't one
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(0)

        x = self.cnn_model(observations)
        x = self.shared_mlp(x)
        action = self.action_layer(x)

        return action
