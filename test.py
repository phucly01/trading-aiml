

class classA:

    one = "One"

    def __init__(self, val):
        self.__class__.one = val
        self.one = 'X'
        

    def show(self, marker):
        print("{}: self.one= {}, self.__class__.one= {}, classA.one= {}, type(self).one = {}".format(marker, self.one, self.__class__.one, classA.one, type(self).one))


one = classA('A')

one.show('One')

two = classA('B')
two.one = 'C'

two.show('Two')
print("{}".format(type(two).one))

three = classA('C')
three.__class__.one = 'F'
classA.one = 'D'
three.one = 'E'
type(three).one = 'G'
three.show('Three')
two.show('Two')
one.show('One')