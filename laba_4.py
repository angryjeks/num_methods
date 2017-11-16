import numpy as np 
import matplotlib.pyplot as plt 
#import seaborn
#import sys
#sys.path.insert(0, 'C:/Users/User/Python/workspace/my_bot')
from num_meth_2 import sign

np.set_printoptions(linewidth = 120)

N = 8
a, b = 0, 2
y = 2*np.exp(2)/3
y_max = np.exp(2)


class Interpolation:

	def __init__(self, a, b, N, f, nodes_type = 'equal'):
		if nodes_type == 'equal':
			self.__x = np.linspace(a, b, N)
		elif nodes_type == 'cheb':
			self.__x = np.array([(a+b)/2 - ((b-a)/2)*cheb(k,N) for k in range(N)])
		else:
			print("wrong type of nodes")

		self.__y = np.array([f(x) for x in __x])


class Polinom:
	
	def __init__(self, coefs, x_list):
		self.__coefs = coefs
		self.__x_list = x_list

	def _evaluate(self, x0):
		_sum = self.__coefs[0]
		for i in range(1, N+1):
			_mul = 1
			for j in range(i-1):
				_mul *= (x0 - self.__x_list[j])
			_sum += self.__coefs[i]*_mul
		return _sum

	def _change_coef(self, i, a):
		try:
			self.__coefs[i] += a
		except IndexError:
			print("Wrong index")


def useless_dichotomy(a, b, polinom, y,eps):
    while abs(a - b) > eps:
        x0 = (a+b)/2
        print(x0)
        if sign(polinom._evaluate(x0)-y) == sign(polinom._evaluate(a)-y):
            a = x0
        else:
            b = x0
    print(polinom._evaluate(x0))
    return x0

def reverse_interpolation(x_l, N, y_find):
	y_l = np.array([f(x) for x in x_l])
	coefs = get_coefs(x_l, y_l)
	polinom = Polinom(coefs, x_l)
	x = useless_dichotomy(1.5, 2., polinom, y_find, pow(10, -8))
	print(x)

	

def create_canvas(x_l, y_l, coefs, string, sub):
	plt.subplot(1,2,1+sub)
	plt.scatter(x_l, y_l, c = 'red', label = 'nodes')
	x1_l = np.linspace(a,b,1000)
	y1_l = np.array([polinom(coefs, x_l, x) for x in x1_l])
	y2_l = np.array([f(x) for x in x1_l])
	plt.plot(x1_l, y1_l, c = 'blue', label = r'$P_n (x)$')
	plt.plot(x1_l, y2_l, c = 'green', label = r'$f (x)$')
	plt.plot(x1_l, y1_l-y2_l, c = 'purple', label = r'$f(x) - P_n(x)$')
	plt.title("Number of numbers = " + str(N) + "\n" + string)
	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend()

def interpolation(N, a,b, x_l, string,sub):
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
	return M[0,:]



def main():
	x_cheb = np.array([(a+b)/2 - ((b-a)/2)*cheb(k,N) for k in range(N)])
	x_equal_spaced = np.linspace(a,b,N)
	print(y)
	reverse_interpolation(x_equal_spaced, N, y)
	plt.figure("Interpolation")
	interpolation(N, a, b, x_cheb, "Chebyshev's nodes",0)
	#plt.grid(True)
	interpolation(N, a, b, x_equal_spaced, "Equal spaced nodes",1)
	#plt.grid(True)
	plt.tight_layout()
	plt.savefig("picture.png")
	plt.show()
	

if __name__ == '__main__':
	main()
