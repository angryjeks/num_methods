import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.text as txt

np.set_printoptions(linewidth = 120)

N = 25
a, b = 0, 10

def f(x):
	return np.cos(4*x)

def polinom(coefs, x_l, x):
	_sum = coefs[0]
	for i in range(1, N+1):
		_mul = 1
		for j in range(i-1):
			_mul *= (x - x_l[j])
		_sum += coefs[i]*_mul
	return _sum

def get_coefs(x_l, y_l):
	M = np.zeros((N, N+1))
	M[:,0] = x_l
	M[:,1] = y_l
	for j in range(2, N+1):
		for i in range(N-j+1):
			M[i,j] = (M[i+1,j-1] - M[i, j-1])/(M[i+j-1,0] - M[i,0])

	with open("out_laba3.txt" , "w") as f:
		f.write(str(M))
	return M[0,:]



def main():
	x_l = np.array(np.linspace(a,b,N))
	print(x_l)
	y_l = np.array([f(x) for x in x_l])

	#x_l = np.array([100,200,300,400,500])
	#y_l = np.array([250,300,380,420,380])

	coefs = get_coefs(x_l, y_l)



	plt.scatter(x_l, y_l, c = 'red', label = 'points')
	x1_l = np.linspace(a,b,1000)
	y1_l = [polinom(coefs, x_l, x) for x in x1_l]
	y2_l = [f(x) for x in x1_l]
	plt.plot(x1_l, y1_l, c = 'blue', label = 'polinom')
	plt.plot(x1_l, y2_l, c = 'green', label = 'func')
	plt.title("Number of numbers = " + str(N))
	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()
