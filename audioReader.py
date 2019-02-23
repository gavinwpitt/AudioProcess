import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.fftpack import fftfreq
from scipy.io.wavfile import read

def generateNotes():
	# if A4 = 440, then C0 = 16.35
	cZero = 16.35
	a = 1.059463094359
	notes = [] 
	noteNames = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
	for n in range(0, 128):
		pitch = cZero * ( a**float(n) )
		noteName = noteNames[(n % 12)]
		notes.append((pitch, noteName))

	return notes

def getNoteFromFreq(frequency, notes):
	minDiff = 100
	closetNote = None
	for note in notes:
		tempDiff = abs(frequency - note[0])
		if tempDiff < minDiff:
			minDiff = tempDiff
			closetNote = note

	return closetNote

def getLowestIndexInTopN(signal, N):
	topSignals = np.argsort(transformSignal)
	lowestIndex = topSignals[len(topSignals)-1]
	for x in range(len(topSignals) - 1, len(topSignals) - (N+1), -1):
		if topSignals[x] < lowestIndex:
			lowestIndex = topSignals[x]

	return lowestIndex

sampleRate = 44100.0
secondStep = 1.0/sampleRate
audioFilePath = "./audioFiles/"

# Generate Notes
notes = generateNotes()

for note in notes:
	print(note)

input(">")

# Read in the signal
sr, signal = read(audioFilePath + "mandoChromatic.wav")

plt.plot(signal)
plt.show()

# read through the signal step by step
stepSize = 10000
index = 0
delta = 1.0/sr
while True:
	if index > len(signal):
		break

	plt.cla()
	# Read in piece of signal
	signalChunk = signal[index:index+stepSize]

	# Get time seconds passed
	secs = len(signalChunk)/float(sr)
	t = np.arange(0, secs, delta)

	# Get FFT and find frequency bins
	transformSignal = fft(signalChunk)
	transformSignal = abs(transformSignal)[:len(transformSignal)//2]
	freqs = fftfreq(len(signalChunk), t[1]-t[0])
	freqs = abs(freqs[:len(freqs)//2])


	# Plot?
	plt.plot(freqs, transformSignal)
	#G3 to B6
	plt.xlim(180,2000)
	plt.ylim(0, 1*(10**7))
	plt.pause(0.1)

	lowestIndex = getLowestIndexInTopN(transformSignal, 2)

	if transformSignal[np.argmax(transformSignal)] > 0.1*(10**7):
		fundamentalFreq = freqs[lowestIndex]
		note = getNoteFromFreq(fundamentalFreq, notes)

		if note:
			print(fundamentalFreq)
			print(note)

	# Iterate
	index += stepSize
	input(">")

plt.show()





'''
print(sr)
print(signal)

plt.plot(signal)
plt.show()

delta = 1.0/sr
secs = len(signal)/float(sr)
t = np.arange(0, secs, delta)
freqs = fftfreq(len(signal), t[1]-t[0])
transformSignal = fft(signal)
plt.plot(freqs[:len(freqs)//2], abs(transformSignal)[:len(signal)//2])
plt.xlim(16,1000) #roughly the range of a guitar
plt.show()
'''