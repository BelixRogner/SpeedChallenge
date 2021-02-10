import os
from FrameDataset import FrameDataset
import torch
import torch.nn as nn
import torch.optim as optim
import time
from utils import loadFile, loadLabels, loadStats
import tqdm
from torchsummary import summary
import torchvision

MAX_EPOCHS = 100
PATIENCE = 5

HEIGHT = 100
WIDTH = 640
SPLITS = 50

BATCH_SIZE = 16
WEIGHT_DECAY = 0.001

CONFIG = {}

CONFIG["seqLen"] = 9

CONFIG["stopPct"] = 0.04
CONFIG["doublePct"] = 0.3
	


print(CONFIG, BATCH_SIZE, WEIGHT_DECAY)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()
criterionValidate = nn.MSELoss(reduction='sum')

frames = loadFile("data/train.mp4", HEIGHT, WIDTH)
stats = loadStats(f"data/train.mp4_modified_{HEIGHT}_{WIDTH}_96.pt")
y = loadLabels("data/train.txt")
pb = tqdm.tqdm()

def trainEpochs(trainloader, validateloader, j):
	lastFile = ""
	bestLoss = float("inf")
	bestEpoch = 0
	net = torchvision.models.video.r2plus1d_18(num_classes=1)
	net.to(device)
	summary(net, input_size=(3, CONFIG["seqLen"], HEIGHT, WIDTH))
	optimizer = optim.AdamW(net.parameters(), lr=5e-4, weight_decay=WEIGHT_DECAY)   
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.93, verbose=False)
	epoch = 0
	
	while epoch < MAX_EPOCHS and epoch - bestEpoch < PATIENCE:
		epoch += 1
		running_loss = 0.0
		count = 0
		net.train()
		pb.reset(len(trainloader))
		pb.set_description(f"Train epoch {epoch}")
		for data in trainloader:
			# get the inputs; data is a list of [inputs, labels]
			inputs = data[0].to(device)
			labels = data[1].to(device).view(-1,1)

			#print(inputs[0], labels[0])
			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			count += 1
			pb.update()
					
		l = running_loss / count
		pb.write(f"train loss of epoch {epoch}: {l:.4f}")
		scheduler.step()

		pb.reset(len(validateloader))
		pb.set_description(f"Validating")
		total = 0
		net.eval()
		outputSave = []
		with torch.no_grad():
			for data in validateloader:
				inputs = data[0].to(device)
				labels = data[1].to(device).view(-1,1)
				outputs = torch.nn.functional.relu(net(inputs), inplace=True)
				outputSave.append(outputs)
				total += criterionValidate(outputs, labels).item()
				pb.update()
			
			loss = total/len(validate)
			pb.write(f"Validate loss: {loss:.4f}")
			
		net.train()
		
		if loss < bestLoss:
			bestLoss = loss
			bestEpoch = epoch
			timeS = time.time()
			fileName = f'results/{loss}-{timeS}-{j}-{SPLITS}'
			torch.save(net, fileName + '.pt')
			with open(fileName + '.txt', 'w') as f:	
				for item in outputSave:
					for x in item:
						f.write(f"{x.item()}\n")

			if lastFile != "":
				os.remove(lastFile + '.txt')
				os.remove(lastFile + '.pt')
			lastFile = fileName
					
	return bestLoss

while True:
	for j in range(5):
		trainSplits = []
		validSplits = []
		for i in range(SPLITS):
			if i % 5 == j:
				validSplits.append(i)
			else:
				trainSplits.append(i)
		trainSplits = torch.tensor(trainSplits)
		validSplits = torch.tensor(validSplits)

		train = FrameDataset(frames, stats, y, train=True, seqLen=CONFIG["seqLen"], doublePct=CONFIG["doublePct"], stopPct=CONFIG["stopPct"], numSplits=SPLITS, splitIdxs=trainSplits)
		validate = FrameDataset(frames, stats, y, train=False, seqLen=CONFIG["seqLen"], numSplits=SPLITS, splitIdxs=validSplits)
		print(len(validSplits), len(trainSplits))
		trainloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, prefetch_factor=1)
		validateloader = torch.utils.data.DataLoader(validate, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, prefetch_factor=1)
		print(trainEpochs(trainloader, validateloader, j))
	