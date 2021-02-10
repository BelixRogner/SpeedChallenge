import torch
from torch.utils.data import Dataset
from torchvision import transforms
from einops import repeat

class FrameDataset(Dataset):
    
    def __init__(self, frames, stats, labels, train=True, seqLen=20, numSplits=10, splitIdxs = [], doublePct=0.0, stopPct=0.0):
        self.normalize = transforms.Normalize(stats[0], stats[1], inplace=True)

        self.augment = transforms.RandomHorizontalFlip()

        splitIdxs = torch.sort(splitIdxs)[0]

        self.doublePct = doublePct
        self.stopPct = stopPct
        self.train = train
        self.seqLen = seqLen
        self.labels = labels
        self.frames = frames
        assert len(self.frames) == len(self.labels)
        if len(self.frames) % numSplits != 0:
            numSplits+=1
        self.splitSize = len(self.frames) // numSplits
        self.splits = torch.zeros(len(splitIdxs), dtype=torch.int32)

        for i in range(len(splitIdxs)):
            self.splits[i] = (splitIdxs[i] - i) * self.splitSize
        #print(self.train, self.splits)

    def __len__(self):
        return len(self.splits) * self.splitSize - self.seqLen + 1

    def __getitem__(self, idx):
        assert not torch.is_tensor(idx)
        idx += self.splits[idx // self.splitSize]

        if self.train:
            rand = torch.rand(1).item()
            if rand < self.stopPct:
                frames = self.normalize(self.frames[idx] / 255)
                frames = repeat(frames, 'c h w -> k c h w', k=self.seqLen)
                label = torch.tensor(0.0)
            elif rand < self.stopPct + self.doublePct and idx + self.seqLen * 2 - 1 <= len(self.frames):
                frames = self.normalize(self.frames[idx : idx + self.seqLen * 2 : 2] / 255)
                label = self.labels[idx + self.seqLen - 1] * 2
            else:
                frames = self.normalize(self.frames[idx : idx + self.seqLen] / 255)
                label = self.labels[idx + self.seqLen // 2]

            frames = self.augment(frames)

        else:
            frames = self.normalize(self.frames[idx : idx + self.seqLen] / 255)
            label = self.labels[idx + self.seqLen // 2]
        return frames.permute(1,0,2,3), label