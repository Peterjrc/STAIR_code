import torch.nn as nn
import torch
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, feature_dim = 10, num_classes = 2):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50,num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc4(x)
        return out


class Model_cleanlab(nn.Module):
    def __init__(self, feature_dim = 10, num_classes = 2):
        super(Model_cleanlab, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100,50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50,num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc4(x)
        return F.softmax(out, dim=1)


class Contrastive_Model(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super(Contrastive_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 8),  # nn.Linear(feature_dim, 8)
            nn.ReLU(),
            nn.Linear(8, 16),  # nn.Linear(8, 16)
            nn.ReLU(),
            nn.Linear(16, 8),  # nn.Linear(16, 8)
            nn.ReLU(),
            nn.Linear(8, self.hidden_dim)  # nn.Linear(8, 128)
        )

    def forward(self, x):
        output = self.fc(x)
        return output


# class Feature_Attention(nn.Module):
#     def __init__(self, feature_dim):
#         super(Feature_Attention, self).__init__()
#         self.feature_dim = feature_dim
#         self.fc = nn.Sequential(
#             nn.Linear(feature_dim, self.feature_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         attention_weights = self.fc(x)
#         return attention_weights


class Reconstruct_Model(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256, z_dim = 128, en_nlayers=3, de_nlayers=3):
        super(Reconstruct_Model, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.en_nlayers = en_nlayers
        self.de_nlayers = de_nlayers

        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.Softmax(dim=1)
        )

        encoder_dim = self.feature_dim
        encoder = []
        for _ in range(self.en_nlayers-1):
            encoder.append(nn.Linear(encoder_dim, self.hidden_dim, bias= False))
            encoder.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = self.hidden_dim

        encoder.append(nn.Linear(encoder_dim, self.z_dim, bias=False))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        decoder_dim = self.z_dim
        for _ in range(self.de_nlayers - 1):
            decoder.append(nn.Linear(decoder_dim, self.hidden_dim, bias=False))
            decoder.append(nn.LeakyReLU(0.2, inplace=True))
            decoder_dim = self.hidden_dim

        decoder.append(nn.Linear(decoder_dim, self.feature_dim, bias=False))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        w = self.attention(x)
        x = x * w
        z = self.encoder(x)
        x_pred = self.decoder(z)
        return x_pred