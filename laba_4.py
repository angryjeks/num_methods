import numpy as np 
import matplotlib.pyplot as plt 
#import seaborn
#import sys
#sys.path.insert(0, 'C:/Users/User/Python/workspace/my_bot')
from num_meth_2 import sign

np.set_printoptions(linewidth = 120)

N = 10
a, b = 0, 2
y = 2*np.exp(2)/3
y_max = np.exp(2)
EPS = pow(10, -10)
polinom_str = '$P_n (x)$'
f_str = '$f (x)$'
polinom_f = '$f(x) - P_n (x)$'

def f(x):
	return np.cos(np.pi*x)**2*np.exp(x)

class Interpolation:

	def __init__(self, a, b, N, func, nodes_type = 'equal'):
		self.__a = a
		self.__b = b
		self._func = func
		self.__string = nodes_type
		if nodes_type == 'equal':
			self.__x = np.array([a + ((b-a)*k)/(N-1) for k in range(N)])
			self.__string = 'Equal nodes'
		elif nodes_type == 'cheb':
			self.__x = np.array([(a+b)/2 - ((b-a)/2)*cheb(k,N) for k in range(N)])
			self.__string = "Chebyshev's nodes"
		else:
			print("wrong type of nodes")
		
		self.__y = np.array([func(x) for x in self.__x])
		self.__N = N
		self.__coefs = self._create_coefs(self.__N)
		self.__polinom = Polinom(self.__coefs, self.__x)

	def __call__(self, x0):
		return self.__polinom._evaluate(x0)

	def __str__(self):
		string = "x = {}\ny = {}\nN = {}".format(self.__x, self.__y, self.__N)
		return string

	def _create_coefs(self, N):
		M = np.zeros((N, N+1))
		M[:,0] = self.__x
		M[:,1] = self.__y
		for j in range(2, N+1):
			for i in range(N-j+1):
				M[i,j] = (M[i+1,j-1] - M[i, j-1])/(M[i+j-1,0] - M[i,0])
		return M[0,:]
	
	def create_canvas(self, sub):
		plt.subplot(1,2,1+sub)
		plt.scatter(self.__x, self.__y, c = 'red', label = 'nodes')
		x1_l = np.linspace(self.get_a(), self.get_b(),1000)
		y1_l = np.array([self(x) for x in x1_l])
		y2_l = np.array([self._func(x) for x in x1_l])
		plt.plot(x1_l, y1_l, c = 'blue', label = r''+polinom_str)
		plt.plot(x1_l, y2_l, c = 'green', label = r''+f_str)
		plt.plot(x1_l, y1_l-y2_l, c = 'purple', label = r''+polinom_f)
		plt.title("Number of numbers = " + str(N) + "\n" + self.get_string())
		plt.xlabel("x")
		plt.ylabel("y")
		plt.legend()

	def reverse_interpolation(self, y0):
		x = useless_dichotomy(1.5, 2., self.__polinom, y0, EPS)
		return x

	#SETTERS AND GETTERS
	def set_x(self, value):
		self.__x = value
	def get_x(self):
		return self.__x
	def set_y(self, value):
		self.__y = value
	def get_y(self):
		return self.__y
	def set_N(self, value):
		self.__N = value
	def get_N(self):
		return self.__N
	def set_coefs(self, value):
		self.__coefs = value
	def get_coefs(self):
		return self.__coefs
	def get_a(self):
		return self.__a
	def get_b(self):
		return self.__b
	def get_string(self):
		return self.__string

	x = property(get_x, set_x)
	y = property(get_y, set_y)
	N = property(get_N, set_N)
	coefs = property(get_coefs, set_coefs)

class Cheb_nodes_interpolation(Interpolation):
	
	def __init__(self, a, b, N, func):
		Interpolation.__init__(self, a, b, N, func, 'cheb')

class Equal_nodes_interpolation(Interpolation):
	
	def __init__(self, a, b, N, func):
		Interpolation.__init__(self, a, b, N, func)

class Polinom:
	
	def __init__(self, coefs, x_list):
		self.__coefs = coefs
		self.__x = x_list

	def _evaluate(self, x0):
		_sum = self.__coefs[0]
		#print(self.__coefs)
		for i in range(1, N+1):
			_mul = 1
			for j in range(i-1):
				_mul *= (x0 - self.__x[j])
			_sum += self.__coefs[i]*_mul
		return _sum

	def _change_coef(self, i, a):
		try:
			self.__coefs[i] += a
		except IndexError:
			print("Wrong index")

	def __call__(self, x0):
		return self._evaluate(x0)

def useless_dichotomy(a, b, polinom, y,eps):
    while abs(a - b) > eps:
        x0 = (a+b)/2
        if sign(polinom(x0)-y) == sign(polinom(a)-y):
            a = x0
        else:
            b = x0
    print(polinom._evaluate(x0))
    return x0

def cheb(k, n):
	return np.cos(((2*k+1)*np.pi)/(2*n))

def main():
	cheb1 = Cheb_nodes_interpolation(a, b, N, f)
	equal = Equal_nodes_interpolation(a, b, N, f)
	plt.figure("Interpolation")
	cheb1.create_canvas(0)
	equal.create_canvas(1)
	x = equal.reverse_interpolation(y)
	print(x, f(x), y)
	plt.tight_layout()
	plt.savefig("picture.png")
	plt.show()
	

if __name__ == '__main__':
	main()
