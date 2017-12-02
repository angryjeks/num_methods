from num_meth_2 import matrix_mul, build_matrix, norm_vector_fro, vector_mul, norm_vector
import numpy as np

EPS = 10**(-10)

np.set_printoptions( suppress = False,linewidth = 200)

def dot_product_method(A, N, T = False):
	A1 = A.copy()
	count = 0
	mu = 0
	_norm = 10
	x = np.array([np.random.rand() for i in range(N)])
	#print(x)
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
		mu2, e_vec_mu2,count2= dot_product_method(B, N)
		mu2 = mu1 - mu2
		return mu1, e_vec, mu2, e_vec_mu2, count, find_lambda_min(A1,mu1, N)
	return mu1, e_vec, count

def find_lambda_min(A1, l_max, N):
	A = A1.copy()
	E = np.eye(N)
	B = E - A.dot(A)/(l_max*l_max)
	b_l, a, b = dot_product_method(B, N)
	return abs(l_max)*(1-b_l)**0.5

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
		sum_sq = 0
		for i in range(A1.shape[0]):
			for j in range(i+1, A1.shape[1]):
				sum_sq += A1[i,j]*A1[i,j]
		for i in reversed(range(A1.shape[0])):
			for j in range(i):
				sum_sq += A1[i,j]*A1[i,j]
		print(sum_sq)
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
	return A1, H1, count, _max_lambda

def main():
	N = 8
	m = 8
	A, b = build_matrix(N,m)
	_lambda, vec1, _lambda_min, vec2, count_dot_product,_l_min_abs = dot_product_method(A,N,True)
	A1_jacobi, H1_jacobi, count_jacobi, _max_lambda = jacobi_eigenvalue(A,N)
	for i in range(len(H1_jacobi)):
		pass
		#print("l[{0}] = {1:.10f},  H[{2}] = ".format(i, A1_jacobi[i,i], i), H1_jacobi[:,i])
	with open("output_laba3.txt", 'w') as f:
		f.write("A = \n" + str(A)+"\n")
		f.write("Jacobi eigenvalue method results: \n" + "A1 = \n" \
			    + str(A1_jacobi) + "\n")
		for i in range(len(H1_jacobi)):
			f.write(("l[{0}] = {1:.10f},  H[{2}] = {3}\n".format(i, A1_jacobi[i,i], i, H1_jacobi[:,i])))
		f.write("count of iterations = " + str(count_jacobi) + "\n")
		f.write("max lambda by Jacobi method = " + str(_max_lambda)+"\n")
		f.write("______________________________________________________\n")
		"""f.write("Dot product method results: \n" + "max |eigenvalue| = " + str(_lambda) + ", vec = " + str(vec1) +"\n"\
		        + "min eigenvalue = " + str(_lambda_min) +  ", vec = " + str(vec2) + "\n"\
		        +"count of iterations = " + str(count_dot_product) + "\n")"""
		f.write("Dot product method results: \n" + "max |eigenvalue| = " + str(_lambda) + "\n"\
		        + "min eigenvalue = " + str(_lambda_min) +  "\n"\
		        + "min |eigenvalue| = " + str(_l_min_abs) +  "\n"\
		        +"count of iterations = " + str(count_dot_product) + "\n")
		f.write("EPS = {}".format(EPS))
	input()


if __name__ == '__main__':
	main()