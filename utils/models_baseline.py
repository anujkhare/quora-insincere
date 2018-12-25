import numpy as np
import torch
import torch.nn.functional as F


class CBOW(torch.nn.Module):
    def __init__(self, vector_dim=300, n_out=2) -> None:
        super().__init__()
        in_size = vector_dim
        
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(in_size, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 512),
            torch.nn.Tanh(),
        )
        self.classifier = torch.nn.Linear(512, n_out)
        self.vector_dim = vector_dim

    def forward(self, data):
        vectors1 = data
        s1 = torch.sum(vectors1, dim=1)
        
        features = self.feature_extractor(s1)
        pred = self.classifier(features)
        return F.log_softmax(pred, dim=1)

    
def test_cbow():
    bsz = 1
    seq1 = torch.from_numpy(np.random.rand(bsz, 20, 300,).astype(np.float32))

    model = CBOW()

    outputs = model(seq1)
    assert outputs.shape == (bsz, 2)

test_cbow()


class LSTM(torch.nn.Module):
    def __init__(self, n_out=2, vector_dim=300, proj_dim=300, num_layers=1, birnn=True) -> None:
        super().__init__()
        
        DIM = 512
        
        self.projection = torch.nn.Linear(vector_dim, proj_dim)
        self.lstm = torch.nn.LSTM(proj_dim, proj_dim, num_layers=num_layers, batch_first=True, bidirectional=birnn)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(proj_dim * num_layers * (int(birnn) + 1), DIM),
            torch.nn.Linear(DIM, DIM),
            torch.nn.Linear(DIM, n_out),
        )
    
    def _sentence_features(self, vectors):
        proj = self.projection(vectors)
        _, (hT, _) = self.lstm(vectors)

        sentence_vec = hT.transpose(0, 1)  # get batch first: B*num_layers*num_directions*300
        sentence_vec = sentence_vec.contiguous().view(sentence_vec.shape[0], -1)
        return sentence_vec

    def forward(self, data):
        vectors1 = data

        features = self._sentence_features(vectors1)
        
        pred = self.classifier(features)
        return F.log_softmax(pred, dim=1)


def test_lstm():
    bsz = 3
    seq1 = torch.from_numpy(np.random.rand(bsz, 20, 300,).astype(np.float32))
    model = LSTM()

    outputs = model(seq1)
    assert outputs.shape == (bsz, 2)

test_lstm()