from num_meth_2 import matrix_mul, build_matrix, norm_vector_fro, vector_mul
import numpy as np

EPS = 10**(-8)

np.set_printoptions( suppress = True,linewidth = 120)

def dot_product_method(A, N, T = False):
	A1 = A.copy()
	count = 0
	mu = 0
	_norm = 10
	x = np.array([-2 for i in range(N)])
	e_vec = x/norm_vector_fro(x)
	while _norm > EPS:
		count+=1
		x = matrix_mul(A1, e_vec)
		mu1 = vector_mul(x, e_vec)
		e_vec = x/norm_vector_fro(x)
		_norm = abs(mu1 - mu)
		mu = mu1
	if T:
		B = mu1*np.eye(N) - A1
		mu2, count2= dot_product_method(B, N)
		mu2 = mu1 - mu2
		return mu1,mu2, count

	return mu1, count

def find_max_element_up_matrix(A,N):
	_max = 0
	_max_pair = (0, 0)
	for i in range(0, N):
		for j in range(i+1, N):
			if abs(A[i,j])>=_max:
				_max = abs(A[i,j])
				_max_pair = (i,j)
	return _max, _max_pair

def jacobi_eigenvalue(A,N):
	_max_lambda = 0
	count = 1
	A1 = A.copy()
	H = np.eye(N)
	H1 = np.eye(N)
	_max, _max_pair = find_max_element_up_matrix(A1,N)
	while abs(_max) > EPS:
		count +=1
		phi = 0.5*(np.arctan((2*A1[_max_pair[0], _max_pair[1]])/(A1[_max_pair[0], _max_pair[0]]-A1[_max_pair[1], _max_pair[1]])))
		sin_phi, cos_phi = np.sin(phi), np.cos(phi)
		H[_max_pair[0],_max_pair[0]] = cos_phi
		H[_max_pair[0],_max_pair[1]] = -sin_phi
		H[_max_pair[1],_max_pair[0]] = sin_phi
		H[_max_pair[1],_max_pair[1]] = cos_phi
		H1 = matrix_mul(H1,H)
		A1 = matrix_mul(matrix_mul(H.transpose(), A1), H)
		H = np.eye(N)
		_max, _max_pair = find_max_element_up_matrix(A1,N)
	for i in range(N):
		if A1[i,i] >= _max_lambda:
			_max_lambda = A1[i,i]
	#print(A1)
	#print("count of iterations = ", count)
	#print(H1)
	return A1, H1, count, _max_lambda

def main():
	N = 10
	m = 10
	A, b = build_matrix(N,m)
	_lambda, _lambda_min,count_dot_product = dot_product_method(A,N,True)
	A1_jacobi, H1_jacobi, count_jacobi, _max_lambda = jacobi_eigenvalue(A,N)
	with open("output_laba3.txt", 'w') as f:
		f.write("A = \n" + str(A)+"\n")
		f.write("Jacobi eigenvalue method results: \n" + "A1 = \n" + str(A1_jacobi) + "\n" + "H1 = \n" + str(H1_jacobi) + "\n" +"count of iterations = " + str(count_jacobi) + "\n")
		f.write("max lambda by Jacobi method = " + str(_max_lambda)+"\n")
		f.write("______________________________________________________\n")
		f.write("Dot product method results: \n" + "max |eigenvalue| = " + str(_lambda) + "\n" + "min |eigenvalue| = " + str(_lambda_min) + "\n" +"count of iterations = " + str(count_dot_product) + "\n")

if __name__ == '__main__':
	main()