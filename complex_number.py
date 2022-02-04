import inspect

class Complex:
    def __init__(self,real,imag):
        self.real = None
        self.imag = None
        self.set_real(real)
        self.set_imag(imag)

    def set_real(self,real):
        self.real=real

    def set_imag(self,imag):
        self.imag=imag

    def __repr__(self):
        return "{0}+{1}j".format(self.real,self.imag)

    def __add__(self,another):
        return Complex(self.real+another.real,self.imag+another.imag)

x=Complex(1,2)
y=Complex(2,4)

# y.set_real = function(x): print(x)
y.set_real(300)
print(x+y) # + operator works now


a = complex(3,4)
print(inspect.getmembers(x))

