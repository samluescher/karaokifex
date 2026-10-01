"""RMVPE, the robust model for vocal pitch estimation in polyphonic music (Wei et al., 2023), vendored for
melody.py: its network as RVC has it (github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI,
infer/rmvpe.py: BiGRU to E2E, unchanged), so its trained weights (rmvpe.pt) load as they are. The mel front end
and the runner below are ours, the same sums as RVC's without its device and CUDA-graph plumbing.

  Rmvpe(weights, device).pitch(audio_16k) -> (f0 Hz a 10 ms frame, 0 where unvoiced; its confidence 0-1)

    MIT License

    Copyright (c) 2023 liujing04
    Copyright (c) 2023 源文雨
    Copyright (c) 2023 Ftps

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""
# ruff: noqa
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        # self.shortcut:Optional[nn.Module] = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        else:
            return self.conv(x) + self.shortcut(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels, out_channels, kernel_size, n_blocks, momentum=momentum
                )
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for i, layer in enumerate(self.layers):
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class ResEncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01
    ):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i, conv in enumerate(self.conv):
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class Intermediate(nn.Module):  #
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(
            ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        for i in range(self.n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i, conv2 in enumerate(self.conv2):
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x) :
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(E2E, self).__init__()
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * nn.N_MELS, nn.N_CLASS), nn.Dropout(0.25), nn.Sigmoid()
            )

    def forward(self, mel):
        # print(mel.shape)
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        # print(x.shape)
        return x


# --- ours: the mel front end RVC's RMVPE uses (16 kHz, 128 bands 30 Hz-8 kHz, 10 ms hop), and a chunked runner ---

SR = 16000
HOP = 160            # 10 ms
WINDOW = 4096        # frames the network sees at once (41 s), a multiple of 32
CONTEXT = 256        # frames of context either side of a window, dropped from its output


class Rmvpe:
    def __init__(self, weights, device: str = "cuda") -> None:
        from librosa.filters import mel
        self.device = torch.device(device)
        basis = mel(sr=SR, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
        self.mel_basis = torch.from_numpy(basis).float().to(self.device)
        self.window = torch.hann_window(1024).to(self.device)
        model = E2E(4, 1, (2, 2))
        model.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
        self.model = model.eval().float().to(self.device)
        cents = 20 * np.arange(360) + 1997.3794084376191
        self.cents = np.pad(cents, (4, 4))

    def mel(self, audio: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(self.device)
        spec = torch.stft(x, n_fft=1024, hop_length=HOP, win_length=1024, window=self.window, center=True,
                          return_complex=True).abs()
        return torch.log(torch.clamp(self.mel_basis @ spec, min=1e-5))       # (128, frames)

    @torch.no_grad()
    def salience(self, audio: np.ndarray) -> np.ndarray:
        """(frames, 360): how likely each 20-cent bin is the voice's pitch, a frame each 10 ms."""
        mel = self.mel(audio)
        n = mel.shape[-1]
        out = np.zeros((n, 360), dtype=np.float32)
        for start in range(0, n, WINDOW):
            a, b = max(0, start - CONTEXT), min(n, start + WINDOW + CONTEXT)
            chunk = mel[:, a:b]
            pad = (-chunk.shape[-1]) % 32
            if pad:
                chunk = F.pad(chunk, (0, pad))
            hidden = self.model(chunk.unsqueeze(0))[0, : b - a].float().cpu().numpy()
            keep = min(n, start + WINDOW)
            out[start:keep] = hidden[start - a : keep - a]
        return out

    def pitch(self, audio: np.ndarray, threshold: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
        """The voice's f0 in Hz each 10 ms (0 unvoiced) and its confidence: RVC's local average of cents
        round the likeliest bin, vectorised."""
        sal = self.salience(audio)
        centre = sal.argmax(1)
        padded = np.pad(sal, ((0, 0), (4, 4)))
        idx = centre[:, None] + np.arange(9)[None, :]
        weights = np.take_along_axis(padded, idx, 1)
        cents = (weights * self.cents[idx]).sum(1) / np.maximum(weights.sum(1), 1e-9)
        confidence = sal.max(1)
        cents[confidence <= threshold] = 0
        f0 = 10 * 2 ** (cents / 1200)
        f0[cents == 0] = 0
        return f0.astype(np.float32), confidence.astype(np.float32)
