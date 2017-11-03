from num_meth_2 import matrix_mul, build_matrix, norm_vector, vector_mul
import numpy as np

EPS = 10**(-4)

np.set_printoptions( suppress = True,linewidth = 120)

def dot_product_method(A, N):
	A1 = A.copy()
	count = 0
	mu = 0
	_norm = 10
	x = np.array([-2 for i in range(N)])
	e_vec = x/norm_vector(x)
	while _norm > EPS:
		count+=1
		x = matrix_mul(A1, e_vec)
		mu1 = vector_mul(x, e_vec)
		e_vec = x/norm_vector(x)
		print(e_vec, mu1)
		_norm = abs(mu1 - mu)
		mu = mu1
	print(mu1, count)
	return mu1

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
	count = 1
	A1 = A.copy()
	H = np.eye(N)
	H1 = np.eye(N)
	_max, _max_pair = find_max_element_up_matrix(A1,N)
	while abs(_max) > EPS:
	#for q in range(5):
		count +=1
		#print(_max_pair)
		phi = 0.5*(np.arctan((2*A1[_max_pair[0], _max_pair[1]])/(A1[_max_pair[0], _max_pair[0]]-A1[_max_pair[1], _max_pair[1]])))
		sin_phi, cos_phi = np.sin(phi), np.cos(phi)
		#print(sin_phi, cos_phi)
		H[_max_pair[0],_max_pair[0]] = cos_phi
		H[_max_pair[0],_max_pair[1]] = -sin_phi
		H[_max_pair[1],_max_pair[0]] = sin_phi
		H[_max_pair[1],_max_pair[1]] = cos_phi
		H1 = matrix_mul(H1,H)
		#print(H)
		A1 = matrix_mul(matrix_mul(H.transpose(), A1), H)
		#print(A1)
		H = np.eye(N)
		_max, _max_pair = find_max_element_up_matrix(A1,N)
	print(A1)
	print("count of iterations = ", count)
	print(H1)

def main():
	N = 5
	m = 5
	"""A = np.array([[5,1,2],
				  [1,4,1],
				  [2,1,3]])"""
	#print(A)
	A, b = build_matrix(N,m)
	a = dot_product_method(A,N)
	jacobi_eigenvalue(A,N)

if __name__ == '__main__':
	main()