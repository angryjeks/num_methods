import numpy as np 
from math import pow

N = 4

def sign(x):
	if x>0:
		return 1
	else:
		if x == 0:
			return 0
		else:
			return -1

def gauss_rev(A, b, TF):
	#print(A)
	x = np.array([0. for i in range(N)])
	if TF == False:
		sum1 = 0
		for i in range(N):
			for j in range(i):
				sum1 -= (x[j]*A[i,j])
			x[i] = (sum1 + b[i])/A[i,i]
			#print(sum + b[i], A[i,i])
			sum1 = 0
	else:
		sum1 = 0
		for i in reversed(range(N)):
			for j in range(N):
				if j == i:
					continue
				sum1 -= (x[j]*A[i,j])
			x[i] = (sum1 + b[i])/A[i,i]
			#print(sum + b[i], A[i,i])
			sum1 = 0
	return x


np.set_printoptions(threshold=np.nan)

A = np.matrix([[1,-1,1,-1],[-1,5,-3,3],[1,-3,-7,1],[-1,3,1,10]])
b = np.array([2,-4,-18,-5])
S = np.zeros((N,N))
D = np.zeros((N,N))

m = 0
bs = 0
boll = True

for i in range(N):
	for j in range(i, N):
		if boll:	
			for q in range(0,i):
				m += pow(np.abs(S[q,i]),2)*D[q,q]
				bs +=(S[q,i]*D[q,q]*S[q,j])
			boll = False
			D[i,i] = sign(A[i,i] - m)
			S[i,i] = pow(np.abs(A[i,i] - m),1/2)
		if j != i:
			bs = 0
			for q in range(0,i):
				bs +=(S[q,i]*D[q,q]*S[q,j])

		if not boll:
			S[i,j] = (A[i,j] - bs)/(D[i,i]*S[i,i])
	#print(m,bs)
	m = 0
	bs = 0
	boll = True
St = S.transpose()
STD = St.dot(D)
#x = np.array([0 for i in range(N)])
print(S)
print(STD)
y = gauss_rev(STD, b, False)
print(y)
x = gauss_rev(S, y, True)
print(x)



#print(A)