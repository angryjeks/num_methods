import numpy as np 
import matplotlib.pyplot as plt 
from num_meth_2 import sign

np.set_printoptions(linewidth = 120)

N = 7
a, b = 0, 2
y = 2*np.exp(2)/3
y_max = np.exp(2)
EPS = pow(10, -10)
polinom_str = 'P_n (x)'
f_str = 'cos(\pi x)^2 e^x'
polinom_f = '{} - {}'.format(f_str, polinom_str)
x_true = 1.84423017466962

def f(x):
	return np.cos(np.pi*x)**2*np.exp(x)


class Interpolation:

	def __init__(self, a, b, N, func, nodes_type = 'equal'):
		self.a = a
		self.b = b
		self.func = func
		self.string = nodes_type
		if nodes_type == 'equal':
			self.x = np.array([a + ((b-a)*k)/(N-1) for k in range(N)])
			self.string = 'Equal nodes'
		elif nodes_type == 'cheb':
			self.x = np.array([(a+b)/2 - ((b-a)/2)*cheb(k,N) for k in range(N)])
			self.string = "Chebyshev's nodes"
		else:
			print("wrong type of nodes")
			return 0
		
		self.y = np.array([func(x) for x in self.x])
		self.N = N
		self.omega = Omega(0, self.x, N)

	def build_omega(self):
		plt.figure('Omega for {}'.format(self.string))
		x_l = np.linspace(self.a, self.b, 1000)
		y_l = [self.omega(x) for x in x_l]
		plt.plot(x_l, y_l, c = 'blue', label = r'$\omega _n (x) = \prod _i (x - x_i) $')
		plt.title("Omega for {}".format(self.string))
		plt.legend(loc = 'lower center')
		plt.savefig("omega_{}.png".format("_".join(self.string.split())))

	def _build_polinom(self):
		self.polinom = Polinom(self._create_coefs(self.N), self.x, self.N)

	def __call__(self, x0):
		return self.polinom._evaluate(x0)

	def __str__(self):
		string = "x = {}\ny = {}\nN = {}".format(self.x, self.y, self.N)
		return string

	def _create_coefs(self, N):
		M = np.zeros((N, N+1))
		M[:,0] = self.x
		M[:,1] = self.y
		for j in range(2, N+1):
			for i in range(N-j+1):
				M[i,j] = (M[i+1,j-1] - M[i, j-1])/(M[i+j-1,0] - M[i,0])
		return M[0,1:]
	
	def create_canvas(self, *args):
		plt.subplot(*args)
		plt.scatter(self.x, self.y, c = 'red', label = 'nodes')
		x1_l = np.linspace(self.get_a(), self.get_b(),1000)
		y1_l = np.array([self(x) for x in x1_l])
		y2_l = np.array([self.func(x) for x in x1_l])
		plt.plot(x1_l, y1_l, c = 'blue', label = r'${}$'.format(polinom_str))
		plt.plot(x1_l, y2_l, c = 'green', label = r'${}$'.format(f_str))
		plt.plot(x1_l, y1_l-y2_l, c = 'purple', label = r'${}$'.format(polinom_f))
		plt.title("Number of nodes = " + str(N) + "\n" + self.get_string())
		plt.ylim(-1, 8)
		plt.grid(True)
		plt.xlabel("x")
		plt.ylabel("y")
		plt.legend()

	def reverse_interpolation(self, y0):
		x = useless_dichotomy(1.5, 2., self.polinom, y0, EPS)
		return x

	#SETTERS AND GETTERS
	def set_x(self, value):
		self.x = value
	def get_x(self):
		return self.x
	def set_y(self, value):
		self.y = value
	def get_y(self):
		return self.y
	def set_N(self, value):
		self.N = value
	def get_N(self):
		return self.N
	def set_coefs(self, value):
		self.coefs = value
	def get_coefs(self):
		return self.coefs
	def get_a(self):
		return self.a
	def get_b(self):
		return self.b
	def get_string(self):
		return self.string

	#x = property(get_x, set_x)
	#y = property(get_y, set_y)
	#N = property(get_N, set_N)
	#coefs = property(get_coefs, set_coefs)

class Reverse_interpolation(Interpolation):
	def __init__(self, a, b, N, func, y,nodes_type = 'equal'):
		Interpolation.__init__(self, a, b, N, func, nodes_type)
		#print(self.__dict__)
		self.x, self.y = self.y, self.x
		print("coefs = ",self.x, self.y)
		self.a, self.b = min(self.x), max(self.x)
		self.N = N
		print(self._create_coefs(self.N))
		self.string = "Reverse interpolation, nodes = {}".format(N)
		self._build_polinom()
		self.y_find = y

	def create_canvas(self, *args):
		plt.subplot(*args)
		self.x_find = self(self.y_find)
		self.title = self.get_string() + "\n{}".format("x = {}, y = {}".format(self.x_find, self.y_find))
		plt.scatter(self.y, self.x, c = 'orange', label = 'nodes')
		x1_l = np.linspace(self.get_a(), self.get_b(),1000)
		y1_l = np.array([self(x) for x in x1_l])
		plt.plot(y1_l, x1_l, c = 'blue', label = r'${}$'.format(polinom_str))
		#plt.scatter(y, self(y), c = 'red', label = "y_find")
		plt.scatter(x_true, self.y_find, c = 'red', label = "y_find")
		plt.title(self.title)
		plt.grid(True)
		plt.xlabel("y")
		plt.ylabel("x")
		plt.legend()


class Cheb_nodes_interpolation(Interpolation):
	
	def __init__(self, a, b, N, func):
		Interpolation.__init__(self, a, b, N, func, 'cheb')

class Equal_nodes_interpolation(Interpolation):
	
	def __init__(self, a, b, N, func):
		Interpolation.__init__(self, a, b, N, func)

class Polinom:
	
	def __init__(self, coefs, x_list, N):
		self.coefs = coefs
		self.x = x_list
		self.N = N

	def _evaluate(self, x0):
		_sum = self.coefs[0]
		for i in range(1, self.N):
			_mul = 1
			for j in range(i):
				_mul *= (x0 - self.x[j])
			_sum += self.coefs[i]*_mul
		return _sum

	def _change_coef(self, i, a):
		try:
			self.coefs[i] += a
		except IndexError:
			print("Wrong index")

	def __call__(self, x0):
		return self._evaluate(x0)

class Omega(Polinom):
	def _evaluate(self, x0):
		_mul = 1
		for x in self.x:
			_mul *= (x0 - x)
		return _mul

def useless_dichotomy(a, b, polinom, y,eps):
    while abs(a - b) > eps:
        x0 = (a+b)/2
        if sign(polinom(x0)-y) == sign(polinom(a)-y):
            a = x0
        else:
            b = x0
    return x0

def cheb(k, n):
	return np.cos(((2*k+1)*np.pi)/(2*n))

def main():
	cheb1 = Cheb_nodes_interpolation(a, b, N, f)
	equal = Equal_nodes_interpolation(a, b, N, f)
	cheb1._build_polinom()
	equal._build_polinom()
	plt.figure("Interpolation")
	#cheb1.create_canvas(1,2,1)
	#equal.create_canvas(1,2,2)
	_min = 10
	with open("laba_4_add.txt", 'w') as f1:
		for i in range(2, 30):
			rev = Reverse_interpolation(1.6, 1.9, i, f, y)
			rev._build_polinom()
			x_rev = rev(y)
			if i == 2 or i == 3 or i ==29:
				rev.create_canvas(1,1,1)
			plt.show()
			if abs(f(x_rev) - y) < _min :
				n = i
				eps = abs(f(x_rev) - y)
				_min = eps
			x = cheb1.reverse_interpolation(y)
			f1.write("N = {}\nx = {}\nx_real = {}\nf(x) = {}\npoh = {}\ny = {}\n________________\n".format(i,x_rev, x, f(x_rev), abs(f(x_rev) - y), y))
		f1.write("n = {} eps = {}\n".format(n, eps))
		_min = 10
		for i in range(2, 30):
			rev = Reverse_interpolation(1.5, 2, i, f, y)
			rev._build_polinom()
			x_rev = rev(y)
			if i == 17:
				rev.create_canvas(1,1,1)
			plt.show()
			if abs(f(x_rev) - y) < _min :
				n = i
				eps = abs(f(x_rev) - y)
				_min = eps
			x = cheb1.reverse_interpolation(y)
			f1.write("N = {}\nx = {}\nx_real = {}\nf(x) = {}\npoh = {}\ny = {}\n________________\n".format(i,x_rev, x, f(x_rev), abs(f(x_rev) - y), y))
		f1.write("n = {} eps = {}".format(n, eps))
	x_rev1 = equal.reverse_interpolation(y)
	print("x = {} f(x) = {} |f(x) - f(x_*) = {}".format(x_rev1, f(x_rev1), abs(f(x_rev1) - y)))
	try:
		plt.tight_layout()
	except:
		pass
	plt.savefig("picture.png")
	#plt.show()
	equal.build_omega()
	cheb1.build_omega()
	

if __name__ == '__main__':
	main()
