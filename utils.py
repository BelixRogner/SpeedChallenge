import os
from einops import rearrange
from torchvision import io, transforms
import torch

def loadFile(file, height, width, outputSize=96, splits=100):
    modified = f"{file}_modified_{height}_{width}_{outputSize}.pt"
    if not os.path.isfile(modified):
        print(f"{modified} not found. Creating file...")
        frames = io.read_video(file, pts_unit="sec")[0]
        left = (640 - width)//2
        frames = rearrange(frames[ : , 360 - height : 360, left:left+width], "f h w c -> f c h w")
        finished = []
        for i in range(splits):
            finished.append(transforms.functional.resize(frames[int(round((i/splits)*len(frames))):int(round(((i+1)/splits)*len(frames)))], (outputSize,outputSize)))
        frames = torch.cat(finished).clone()
        torch.save(frames, modified)
        #io.write_video(modified, frames, 20)
    else:
        frames = torch.load(modified)
    
    return frames
    #return io.read_video(modified, pts_unit="sec")[0]

def loadLabels(file):
    with open(file,"r") as data:
        labels = torch.tensor([float(i) for i in data.read().split("\n")])
        return labels

def loadStats(file):
    if not os.path.isfile(file):
        print(f"{file} doesn't exist")
        quit()
    modified = f"{file}_MeanStd.pt"
    if not os.path.isfile(modified):
        print(f"{modified} not found. Creating file...")
        data = torch.load(file) / 255
        stats = torch.stack((torch.mean(data, (0,2,3)), torch.std(data, (0,2,3))))
        torch.save(stats, modified)
    else:
        stats = torch.load(modified)
    return stats