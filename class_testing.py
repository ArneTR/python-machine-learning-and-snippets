class SomeClass:
    variable_1 = " This is a class variable"
    variable_2 = 100    #this is also a class variable.

    def __init__(self, param1, param2):
        self.instance_var1 = param1        #instance_var1 is a instance variable
        self.instance_var2 = param2        #instance_var2 is a instance variable

    def codeMe(self):
        print("CodeMe")

    @classmethod
    def changeClassVar(cls, class_var):
        SomeClass.variable_1 = class_var

x = SomeClass(1,2)
y = SomeClass(2,3)

SomeClass.changeClassVar(30000)
SomeClass.codeMe()


print(x.instance_var1)
print(x.variable_1)

print(y.instance_var1)
print(y.variable_1)