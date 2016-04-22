from scipy import stats
import numpy as np
import random

def linearRegression():
	x = np.array([0.75, 1, 2, 2.25, 2.5, 2.5, 3, 3, 3, 3.25, 3.75, 3.75, 4, 4.5])
	y = np.array([4.5, 2.5, 1, 2.5, 0, 1.5, 1.5, 3, 4, 2.5, 3.75, 4.25, 5, 2.75])

	slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

	print "slope:"
	print slope
	print "intercept:"
	print intercept

def RANSAC():
	x = np.array([0.75, 1, 2, 2.25, 2.5, 2.5, 3, 3, 3, 3.25, 3.75, 3.75, 4, 4.5])
	y = np.array([4.5, 2.5, 1, 2.5, 0, 1.5, 1.5, 3, 4, 2.5, 3.75, 4.25, 5, 2.75])
	k = 2.00
	P = 0.99
	maxInliers = 0
	mostVotedP1 = 0
	mostVotedP2 = 0
	rounds = 3

	for i in xrange(rounds):
		randP1 = random.randrange(0, 14);
		randP2 = random.randrange(0, 14);
		while (randP1 == randP2):
			randP2 = random.randrange(0, 14);
		x1 = x[randP1]
		x2 = x[randP2]
		y1 = y[randP1]
		y2 = y[randP2]
		m = (y2 - y1) / (x2 - x1)
		b = y1 - m * x1
		inliers = 0

		for j in xrange(14):
			x0 = x[j]
			y0 = y[j]
			distance = abs(m * x0 - y0 + b) / np.sqrt(np.power(m, 2) + 1)
			if (distance <= 1):
				inliers += 1

		if (inliers > maxInliers):
			mostVotedP1 = randP1
			mostVotedP2 = randP2
			maxInliers = inliers

	outLiers = 14 - maxInliers
	print "inliers:"
	print maxInliers
	print "outLiers:"
	print outLiers
	print "P1:"
	print np.array([x[mostVotedP1], y[mostVotedP1]])
	print "P2:"
	print np.array([x[mostVotedP2], y[mostVotedP2]])
	print "m and b:"
	bestM = (y[mostVotedP1] - y[mostVotedP2]) / (x[mostVotedP1] - x[mostVotedP2])
	bestB = y[mostVotedP1] - bestM * x[mostVotedP1]
	print bestM, bestB

RANSAC()