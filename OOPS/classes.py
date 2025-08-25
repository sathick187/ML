class Car():
    def __init__(self,name,door):
        self.name=name
        self.door=door

    def met(self):
        print(f"the car name is {self.name}")

class over(Car):
    def __init__(self,name,door):
        Car.__init__(self,name,door)
    def meth(self):
        print(f"the car name is {self.name}") 
a= over("new",5)
a.meth()
print(dir(a))