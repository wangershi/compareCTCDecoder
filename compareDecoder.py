# -*- coding:utf-8 -*-
# Program:
#		test CTC forward algorithm showed \
#		by https://blog.csdn.net/JackyTintin/article/details/79425866
# Release:
#		2018/07/03	ZhangDao	First release

import numpy as np
from collections import defaultdict

nInf = -np.float('inf')

def _logSumExp(a, b):
	'''np.log(np.exp(a) + np.exp(b))
	Args:
		a:	number
		b: 	number
	Returns:
		np.log(np.exp(a)+np.exp(b))
	'''
	# let a >= b
	if a < b:
		a, b = b, a

	if b == nInf:
		return a
	else:
		return a + np.log(1 + np.exp(b - a))

def logSumExp(*args):
	'''from scipy.special import logSumExp
	Args:
		*args:	all input arguments
	Returns:
		np.log(np.exp(arg[0])+...+np.exp(arg[n-1]))
	'''
	res = args[0]
	for e in args[1:]:
		res = _logSumExp(res, e)
	return res

def removeBlank(labels, blank=0):
	'''remove blanks and duplicate elements from labels
	Args:
		labels:	label list with 'int' elements
		black: value of blank, default is zero
	Returns:
		label list without blank
	'''
	newLabels = []

	# combine duplicate
	previous = None
	for l in labels:
		if l != previous:
			newLabels.append(l)
			previous = l

	# remove blank
	newLabels = [l for l in newLabels if l != blank]

	return newLabels

def greedyDecode(y, black=0):
	'''greedy decode
	Args:
		y:		numpy.array with 2 dimensions
				dimension 0 represent 'time'
				dimension 1 represent 'sequence of network outputs'
		blank:	value of blank, default is zero
	Returns:
		rawRs:	raw result
		rs:		result without blanks and duplicate elements
		score:	probability of the sequence
	'''
	rawRs = np.argmax(y, axis=1)
	maxNumber = y[xrange(y.shape[0]), rawRs]
	score = np.multiply.accumulate(maxNumber)[-1]
	rs = removeBlank(rawRs, black)
	return rawRs, rs, score

def beamDecode(y, beamSize=10):
	'''beam decode
	Args:
		y:			numpy.array with 2 dimensions
					dimension 0 represent 'time'
					dimension 1 represent 'sequence of network outputs'
		beamSize:	beam size, default is 10
	Returns:
		beam:		tuple list
					element 0 in tuple represent list without blanks and duplicate elements
					element 1 in tuple represent score
	'''
	T, V = y.shape
	logY = np.log(y)

	beam = [([], 0)]
	for t in range(T):	# for every timestep
		newBeam = []
		for prefix, score in beam:
			for i in range(V):	# for every state
				newPrefix = prefix + [i]

				# log(a * b) = log(a) + log(b)
				newScore = score + logY[t, i]

				newBeam.append((newPrefix, newScore))
		# sort by the score
		newBeam.sort(key=lambda x: x[1], reverse=True)
		beam = newBeam[:beamSize]
	return beam

def prefixBeamDecode(y, beamSize=100, blank=0):
	'''prefix beam decode
	Args:
		y:			numpy.array with 2 dimensions
					dimension 0 represent 'time'
					dimension 1 represent 'sequence of network outputs'
		beamSize:	beam size, default is 10
		blank:		value of blank, default is zero
	Returns:
		beam:		tuple list
					element 0 in tuple represent list without blanks and duplicate elements
					element 1 in tuple is a tuple (probabilityWithBlank, probabilityNoBlank)
	'''
	T, V = y.shape
	logY = np.log(y)

	beam = [(tuple(), (0, nInf))]	# blank, non-blank
	for t in range(T):	# for every timestep
		newBeam = defaultdict(lambda : (nInf, nInf))

		for prefix, (probabilityWithBlank, probabilityNoBlank) in beam:
			for i in range(V):	# for every state
				p = logY[t, i]

				if i == blank:	# propose a blank
					newProbabilityWithBlank, newProbabilityNoBlank = newBeam[prefix]
					newProbabilityWithBlank = logSumExp(newProbabilityWithBlank, probabilityWithBlank + p, probabilityNoBlank + p)
					newBeam[prefix] = (newProbabilityWithBlank, newProbabilityNoBlank)
					continue
				else:	# extend with non-blank
					endT = prefix[-1] if prefix else None

					# extend current prefix
					newPrefix = prefix + (i,)
					newProbabilityWithBlank, newProbabilityNoBlank = newBeam[newPrefix]
					if i != endT:
						newProbabilityNoBlank = logSumExp(newProbabilityNoBlank, probabilityWithBlank + p, probabilityNoBlank + p)
					else:
						newProbabilityNoBlank = logSumExp(newProbabilityNoBlank, probabilityWithBlank + p)
					newBeam[newPrefix] = (newProbabilityWithBlank, newProbabilityNoBlank)

					# keep current prefix
					if i == endT:
						newProbabilityWithBlank, newProbabilityNoBlank = newBeam[prefix]
						newProbabilityNoBlank = logSumExp(newProbabilityNoBlank, probabilityNoBlank + p)
						newBeam[prefix] = (newProbabilityWithBlank, newProbabilityNoBlank)

		# sort by the sum of probabilityWithBlank, probabilityNoBlank
		beam = sorted(newBeam.items(), key=lambda x : logSumExp(*x[1]), reverse=True)
		beam = beam[:beamSize]
	return beam

def solve():
	y = np.array([[0.25, 0.4, 0.35], [0.4, 0.35, 0.25], [0.1, 0.5, 0.4]])
	
	print ('---------------------------------------------')
	print ('raw decode:')
	beam = beamDecode(y, beamSize=27)
	print ('\tBefore many-to-one map:')
	for string, score in beam:
		print ("\tstring=%s\tscore=%.4f" % (string, np.exp(score)))
	# string=[1, 0, 1]	score=0.0800
	# string=[1, 1, 1]	score=0.0700
	# string=[2, 0, 1]	score=0.0700
	# string=[1, 0, 2]	score=0.0640
	# string=[2, 1, 1]	score=0.0612
	# string=[1, 1, 2]	score=0.0560
	# string=[2, 0, 2]	score=0.0560
	# string=[1, 2, 1]	score=0.0500
	# string=[0, 0, 1]	score=0.0500
	# string=[2, 1, 2]	score=0.0490
	# string=[2, 2, 1]	score=0.0437
	# string=[0, 1, 1]	score=0.0437
	# string=[1, 2, 2]	score=0.0400
	# string=[0, 0, 2]	score=0.0400
	# string=[2, 2, 2]	score=0.0350
	# string=[0, 1, 2]	score=0.0350
	# string=[0, 2, 1]	score=0.0312
	# string=[0, 2, 2]	score=0.0250
	# string=[1, 0, 0]	score=0.0160
	# string=[1, 1, 0]	score=0.0140
	# string=[2, 0, 0]	score=0.0140
	# string=[2, 1, 0]	score=0.0123
	# string=[1, 2, 0]	score=0.0100
	# string=[0, 0, 0]	score=0.0100
	# string=[2, 2, 0]	score=0.0087
	# string=[0, 1, 0]	score=0.0087
	# string=[0, 2, 0]	score=0.0063

	print ('\n\tAfter many-to-one map:')
	newBeam = defaultdict(lambda : 0)
	for string, score in beam:
		newBeam[tuple(removeBlank(string))] += np.exp(score)
	beam = sorted(newBeam.items(), key=lambda x : x[1], reverse=True)
	for string, score in beam:
		print ("\tstring=%s\tscore=%.4f" % (string, score))
	# string=(2, 1)	score=0.2185
	# string=(1, 2)	score=0.2050
	# string=(1,)	score=0.2025
	# string=(2,)	score=0.1290
	# string=(1, 1)	score=0.0800
	# string=(2, 2)	score=0.0560
	# string=(1, 2, 1)	score=0.0500
	# string=(2, 1, 2)	score=0.0490
	# string=()	score=0.0100

	print ('\n---------------------------------------------')
	print ('greedy decode:')
	rr, rs, score = greedyDecode(y)
	print ('\tBefore many-to-one map:')
	print ('\t%s' % rr)					# [1 0 1]
	print ('\tAfter many-to-one map:')
	print ('\t%s' % rs)					# [1, 1]
	print ('\tscore is %.4f' % score)	# score is 0.0800

	print ('\n---------------------------------------------')
	print ('beam decode:')
	beam = beamDecode(y, beamSize=2)
	for string, score in beam:
		print ('\tB(%s) = %s, score is %.4f' % (string, removeBlank(string), np.exp(score)))
	# B([1, 0, 1]) = [1, 1], score is 0.0800
	# B([1, 1, 1]) = [1], score is 0.0700

	print ('\n---------------------------------------------')
	print ('prefix beam decode:')
	beam = prefixBeamDecode(y, beamSize=2)
	for string, score in beam:
		print ('\tB(%s) = %s, score is %.4f' % (string, removeBlank(string), np.exp(logSumExp(*score))))
	# B((1, 2)) = [1, 2], score is 0.1200
	# B((2, 1)) = [2, 1], score is 0.1137

if __name__ == '__main__':
	solve()