import numpy as np 
from math import pow

#Num = 4
EPS = pow(10, -8)

def print_matr(A):
	s = ""
	for i in range(np.shape(A)[0]):
		for j in range(np.shape(A)[1]):
			s += str(A[i,j].round(5)) + " | "
		s +="\n"
	return s

def perturbation(A, b, x, N, perturbation):
	A1 = np.copy(A)
	print(A1)
	for i in range(N):
		A1[i,i] = A[i,i]*(1.0 + perturbation)
	print(A1)
	x1 = squares_meth(A1,b,N)
	delta = np.linalg.norm(x1-x)/np.linalg.norm(x)
	delta_matr = np.linalg.norm(A1-A)/np.linalg.norm(A)
	return delta, delta_matr, x1

def print_results(f, A, b,x, N):
	A1 = np.linalg.inv(A)
	E = np.eye(N)
	f.write("A = \n" + str(A)+ "\n")
	f.write("b = \n" + str(b) + "\n")
	f.write("Ax-b = \n" + str(A.dot(x)-b) +"\n")
	f.write("det(A) = "+ str(np.linalg.det(A))+"\n")
	f.write("A^-1 = \n"+ str(A1)+"\n")
	f.write("A^(-1)*A = \n"+ str(A1.dot(A))+"\n")
	f.write("cond(A) = "+ str(np.linalg.cond(A))+"\n")
	f.write("A^(-1)*A - E = " + "\n" + str(A1.dot(A)- E)+"\n")

def det_trian_matr(A):
	mul = 1
	for i in range(np.shape(A)[0]):
		mul *= A[i,i]
	return mul

def to_trian(Ab, N):
	A = Ab
	mul = 1
	for j in range(N-1):
		m = np.eye(N)
		for i in range(j, N):
			if i == j:
				m[i,i] = 1/A[i,i]
				mul*=A[i,i]
			else:
				m[i,j]=-A[i,j]/A[j,j]
		A = m.dot(A)
	#print(A)
	mul*=A[N-1,N-1]
	return A, mul

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
	y = gauss_rev(STD, b, False, N)
	x = gauss_rev(S, y, True, N)
	return x
	
	
def jacobi(A, b, N):
	x0 = np.array([b[i]/A[i,i] for i in range(N)])
	x1 = np.array([0. for i in range(N)])
	count = 0
	while np.linalg.norm(x1-x0)>EPS:
		x0 = x1.copy()
		#print(x0)
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
	y1 = np.array([A[0,j] for j in range(N)])
	return x1, count

def main():
	f = open('output.txt', 'w')
	N = 10
	m = 10
	A = np.zeros((N,N))
	E = np.eye(N)
	b = np.array([pow(i,2) - N for i in range(N)])
	for i in range(N):
		for j in range(N):
			if i != j:
				A[i,j] = (i+j)/(m+N)
			else:
				A[i,i] = N + m + j/m + i/N
	Ab = np.column_stack((A,b))
	gauss_res, detA =to_trian(Ab, N)
	A11 = gauss_res[:,:-1]
	b11 = gauss_res[:,-1]

	x1 = squares_meth(A, b, N)
	x2, count = jacobi(A, b, N)

	delta, delta_matr, x_per = perturbation(A,b,x1,N, 0.1)
	print(delta, x_per, end = '\n')
	print_results(f, A, b, x1, N)
	gauss_res_final = gauss_rev(A11, b11, True, N)

	f.write("perturbation of vectors = " + str(delta) + "\n")
	f.write("perturbation of matrices = " + str(delta_matr) + "\n")
	print("solution by gauss: \n", gauss_res_final)
	print("solution by squares method: \n", x1)
	print("solution by Jacobi method: \n", x2, "\n" , count)
	
	f.write("solution by gauss: \n" +str(gauss_res_final) +"\n")
	f.write("solution by squares method: \n" +str(x1)+"\n")
	f.write("solution by Jacobi method: \n"+str(x2) + "\n" +"count of iterations = " +str(count)+"\n")
	f.close()
	#print(x2-x1)

if __name__ == '__main__':

	main()