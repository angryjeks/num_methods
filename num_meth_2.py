import numpy as np 
from math import pow

#Num = 4
EPS = pow(10, -6)

def sign(x):
	if x>0:
		return 1
	else:
		if x == 0:
			return 0
		else:
			return -1

def gauss_rev(A, b, TF, N):
	x = np.array([0. for i in range(N)])
	if TF == False:
		sum1 = 0
		for i in range(N):
			for j in range(i):
				sum1 -= (x[j]*A[i,j])
			x[i] = (sum1 + b[i])/A[i,i]
			sum1 = 0
	else:
		sum1 = 0
		for i in reversed(range(N)):
			for j in range(i+1, N):
				sum1 -= (x[j]*A[i,j])
			x[i] = (sum1 + b[i])/A[i,i]
			sum1 = 0
	return x


np.set_printoptions(threshold=np.nan)

#A = np.matrix([[1,-1,1,-1],[-1,5,-3,3],[1,-3,-7,1],[-1,3,1,10]])
#b = np.array([2,-4,-18,-5])


def squares_meth(A, b, N):

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
		m = 0
		bs = 0
		boll = True
	St = S.transpose()
	STD = St.dot(D)
	#print(S)
	#print(STD)
	y = gauss_rev(STD, b, False, N)
	#print(y)
	x = gauss_rev(S, y, True, N)
	#print(x)
	return x
	
	
def jacobi(A, b, N):
	x0 = np.array([b[i]/A[i,i] for i in range(N)])
	x1 = np.array([0. for i in range(N)])
	count = 0
	while np.linalg.norm(x1-x0)>EPS:
		x0 = x1.copy()
		x1 = np.array([0. for i in range(N)])
		for i in range(N):
			sum = 0
			for j in range(N):
				if j == i:
					continue
				else:
					sum -=(A[i,j]/A[i,i])*x0[j]
			x1[i] = sum + b[i]/A[i,i]
		count +=1
		#print(x1)
	y1 = np.array([A[0,j] for j in range(N)])

	#print(y1.dot(x1.transpose())-b[0])
	return x1, count

def main():
	N = 25
	m = 10
	A = np.zeros((N,N))
	E = np.eye(N)
	print(E)
	b = np.array([pow(i,2) - N for i in range(N)])
	for i in range(N):
		for j in range(N):
			if i != j:
				A[i,j] = (i+j)/(m+N)
			else:
				A[i,i] = N + m + j/m + i/N
	#print(A)
	#print(b)
	x1 = squares_meth(A, b, N)
	x2, count = jacobi(A, b, N)
	A1 = np.linalg.inv(A)
	print("A = \n", A, "\n","b = \n", b)
	print("Ax-b = \n", A.dot(x1)-b)
	print("det(A) = \n", np.linalg.det(A))
	print("A^-1 = \n", A1)
	print("A^(-1)*A = \n", A1.dot(A))
	print("cond(A) = \n", np.linalg.cond(A))
	print((A1.dot(A)- E))
	#print(np.linalg.solve(A,b))
	print("solution by squares method: \n", x1)
	print("solution by Jacobi method: \n", x2, "\n" , count)
	#print(x2-x1)

if __name__ == '__main__':

	main()