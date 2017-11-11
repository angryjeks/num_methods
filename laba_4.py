import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.text as txt

np.set_printoptions(linewidth = 120)

N = 8
a, b = 0, 2

def create_canvas(x_l, y_l, coefs, string, sub):
	plt.subplot(211+sub)
	plt.scatter(x_l, y_l, c = 'red', label = 'points')
	x1_l = np.linspace(a,b,1000)
	y1_l = [polinom(coefs, x_l, x) for x in x1_l]
	y2_l = [f(x) for x in x1_l]
	plt.plot(x1_l, y1_l, c = 'blue', label = 'polinom')
	plt.plot(x1_l, y2_l, c = 'green', label = 'func')
	plt.title("Number of numbers = " + str(N) + "\n" + string)
	plt.xlabel("x")
	plt.ylabel("y")
	if sub == 1:
		plt.axes([1.,1.,1.,1.])
	plt.legend()
	#plt.show()

def interpolation(N, a,b, x_l, string,sub):
	#x_l = np.array([(a+b)/2 - ((b-a)/2)*cheb(k,N) for k in range(N)])
	y_l = np.array([f(x) for x in x_l])
	coefs = get_coefs(x_l, y_l)
	create_canvas(x_l, y_l, coefs, string,sub)


def cheb(k, n):
	return np.cos(((2*k+1)*np.pi)/(2*n))

def f(x):
	return np.cos(np.pi*x)**2*np.exp(x)

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

	#with open("out_laba4.txt" , "w") as f:
		#f.write(str(M))
	return M[0,:]



def main():
	x_cheb = np.array([(a+b)/2 - ((b-a)/2)*cheb(k,N) for k in range(N)])
	x_equal_spaced = np.linspace(a,b,N)
	plt.figure(1)
	interpolation(N, a, b, x_cheb, "Chebyshev's nodes",0)
	interpolation(N, a, b, x_equal_spaced, "Equal spaced nodes",1)
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	main()
