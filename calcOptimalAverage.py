import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


fileName = "0.189365472205437-1611274226.5689175-2-50.txt"

with open("results/" + fileName, "r") as file:
    results = np.array([float(i) for i in file.read().split("\n")[:-1]])

with open("data/train.txt","r") as file:
    labels = np.array([float(i) for i in file.read().split("\n")])
SEQ_LEN = 9
SPLITS = int(fileName.split("-")[-1].split(".")[0])
offset = SEQ_LEN // 2
chunkSize = 20400 // SPLITS
validSplits = []
for i in range(SPLITS):
	if i % 5 == int(fileName.split("-")[-2]):
		validSplits.append(i)
validSplits = np.array(validSplits)
print(validSplits)
end = np.full(offset*2, results[-1])
results = np.concatenate((results, end))
labels = np.append(labels, np.full(offset, labels[-1]))
#print(labels[:10], results[:10])
y = []
Y = []
for i in range(len(validSplits)):
    startIdx = validSplits[i] * chunkSize
    y.append(results[i * chunkSize : i * chunkSize + chunkSize])
    Y.append(labels[startIdx + offset: startIdx + chunkSize + offset])
    if y[-1].shape != Y[-1].shape:
        print(y[-1].shape, Y[-1].shape)

print(np.mean((np.concatenate(y)-np.concatenate(Y))**2))


low = 99.0
lowIdx = 0
for i in range(1,chunkSize):
    loss = .0
    count = 0
    for block in range(len(Y)):
        count += len(y[block])
        temp = moving_average(y[block], i)
        start = np.full(i//2, temp[0])
        end = np.full(i//2+i%2-1, temp[-1])
        temp = np.concatenate((start, temp, end))
        temp = temp - Y[block]
        loss += np.sum(temp**2)
    loss = loss/count
    #print(i, loss)
    if loss < low:
        lowIdx = i
        low = loss
    if i == 1: print(i, loss)

print(lowIdx, low)

plt.plot(np.concatenate(y), "r")
plt.plot(np.concatenate(Y), "g")
plt.show()

avareged = []
for block in range(len(Y)):
    temp = moving_average(y[block], lowIdx)
    start = np.full(lowIdx//2, temp[0])
    end = np.full(lowIdx//2+lowIdx%2-1, temp[-1])
    temp = np.concatenate((start, temp, end))
    avareged.append(temp)

plt.plot(np.concatenate(avareged), "r")
plt.plot(np.concatenate(Y), "g")
plt.show()