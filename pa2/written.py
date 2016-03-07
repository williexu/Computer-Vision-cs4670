import numpy as np

def written():

	I = [[1,1,0,0,1,1],[1,1,0,0,1,1],[0,0,0,0,1,1],[0,0,0,0,1,1]]
	kernelX = [[0,0,0],[1,-1,0],[0,0,0]]
	kernelY = [[0,1,0],[0,-1,0],[0,0,0]]
	def getIxIy(kernel):
		result = np.empty([4,6])
		for x in xrange(6):
			for y in xrange(4):
				val = 0
				for m in xrange(3):
					m -= 1
					for n in xrange(3):
						n -= 1
						ym = y + m
						xn = x + n
						if (ym < 0):
							ym += 2
						if (ym > 3):
							ym -= 2
						if (xn < 0):
							xn += 2
						if (xn > 5):
							xn -= 2
						val += kernel[-m + 1][-n + 1] * I[ym][xn]
				result[y][x] = val
		return result
	Ix = getIxIy(kernelX)
	Iy = getIxIy(kernelY)
	print "Solution for 1.a:"
	print "Ix:"
	print Ix
	print "Iy:"
	print Iy

	window = [[0,1,0],[1,2,1],[0,1,0]]
	alpha = 0.04
	def getR(window, Ix, Iy):
		R = np.empty([4,6])
		for x in xrange(6):
			for y in xrange(4):
				val1 = 0
				M = np.empty([2,2])
				for m in xrange(3):
					m -= 1
					for n in xrange(3):
						n -= 1
						ym = y + m
						xn = x + n
						if (ym >= 0 and ym < 4 and xn >= 0 and xn < 6):
							M[0][0] += window[m + 1][n + 1] * Ix[ym][xn] * Ix[ym][xn]
							M[0][1] += window[m + 1][n + 1] * Ix[ym][xn] * Iy[ym][xn]
							M[1][0] = M[0][1]
							M[1][1] += window[m + 1][n + 1] * Iy[ym][xn] * Iy[ym][xn]
				R[y][x] = np.linalg.det(M) - alpha * np.trace(M) * np.trace(M)
		return R
	R = getR(window, Ix, Iy)
	print "Solution for 1.b:"
	print "R:"
	print R