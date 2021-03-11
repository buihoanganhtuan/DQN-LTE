"""import matplotlib.pyplot as plt
import os
path = os.path.dirname(__file__)
img_path = '\\figs'
path += img_path
if not os.path.exists(path):
    os.makedirs(path)
#path = os.path.join(os.getcwd(),img_path)
#lt.ion()
#plt.figure()
#plt.plot([1,2,3], label='line1')
#plt.legend()
#plt.draw()
#plt.clf()
plt.figure()
plt.plot([4,1,2], label='line2')
plt.legend()
plt.draw()
print('pass')
a = 1
img_name = '\\name{}.png'.format(1)
plt.savefig(path+img_name)
input('wating user input')"""
"""from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Lambda
import numpy as np
import tensorflow.keras.backend as K

def custom_loss(args):
    y_true, y_pred, mask = args
    return K.sum(y_true, axis=-1, keepdims=False)

model = Sequential()
model.add(Input(shape=(3,)))
model.add(Dense(units=10))
model.add(Dense(units=4))
print(model.summary())

y_pred = model.output
y_true = Input(name='y_true', shape=(4,))
mask = Input(name='mask', shape=(4,))
loss_out = Lambda(custom_loss, output_shape=(1,), name='loss')([y_pred, y_true, mask])
ins = [model.input]
t_model = Model(inputs=ins+[y_true, mask], outputs=[loss_out, y_pred])
print(t_model.summary())
losses = [
    lambda y_true, y_pred : y_pred,
    lambda y_true, y_pred : K.zeros_like(y_pred),
]

a = [np.random.rand(32,3), np.random.rand(32,4), np.random.rand(32,4)]
b = [np.random.rand(32,), np.random.rand(32,4)]

t_model.compile(optimizer='Adam', loss=losses)
t_model.train_on_batch(a,b)

class test(object):
    def __init__(self, a):
        self.a = a
    def method1(self):
        return self.a

class test_advanced(test):
    def __init__(self, b):
        self.b = b
        #super().__init__(b-10)
    def method1(self):
        #super().method1()
        return self.b

t = test_advanced(2)
print(t.method1())


class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

class Square(Rectangle):
    def __init__(self, length):
        super(Square, self).__init__(length, length)

class RightPyramid(Square, Triangle):
    def __init__(self, base, slant_height):
        self.base = base
        self.slant_height = slant_height
        super().__init__(self.base)

    def area(self):
        base_area = super().area()
        perimeter = super().perimeter()
        return 0.5 * perimeter * self.slant_height + base_area

class Triangle:
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return 0.5 * self.base * self.height

class RightPyramid(Triangle, Square):
    def __init__(self, base, slant_height):
        self.base = base
        self.slant_height = slant_height

    def area(self):
        base_area = super().area()
        perimeter = super().perimeter()
        return 0.5 * perimeter * self.slant_height + base_area

pyramid = RightPyramid(2, 4)
pyramid.area()"""

"""class A(object):
    def __init__(self, a):
        self.a = a
    def func_on_a(self, new_a):
        self.modify_a(new_a)
    def modify_a(self, new_a):
        raise NotImplementedError()

class C(A):
    def __init__(self, a):
        super().__init__(a)
    def modify_a(self, new_a):
        self.a = new_a


b = [1,2,3]
ins_a = A(b)
b[0] = 10
print(ins_a.a) # Result: [10, 2, 3]

ins_c = C([4,5,6])
ins_c.modify_a([10, 9])
print(ins_c.a) # Result: [10, 9]"""

# ====================== Modifying input of a function test ====================
class A(object):  
    def modifyInput_type1(self, b):
        b[0] = 100
    def modifyInput_type2(self,b):
        b = [1,1,1]

a = A()
b = [10,1,2]
a.modifyInput_type1(b)
print(b) # Result: [100, 1, 2]

b = [10,1,2]
a.modifyInput_type2(b)
print(b) # Result: [10, 1, 2]

"""
from keras.optimizers import Optimizer
from keras import backend as K
import numpy as np
if K.backend() == 'tensorflow':
    import tensorflow as tf
class COCOB(Optimizer):
    """"""Coin Betting Optimizer from the paper:
        https://arxiv.org/pdf/1705.07795.pdf
    """"""
def __init__(self, alpha=100, **kwargs):
        """"""
        Initialize COCOB Optimizer
        Args:
            alpha: Refer to paper.
        """"""
        super(COCOB, self).__init__(**kwargs)
        self._alpha = alpha
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
def get_updates(self, params, loss, contraints=None):
        self.updates = [K.update_add(self.iterations, 1)]
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        L = [K.variable(np.full(fill_value=1e-8, shape=shape)) for shape in shapes]
        reward = [K.zeros(shape) for shape in shapes]
        tilde_w = [K.zeros(shape) for shape in shapes]
        gradients_sum = [K.zeros(shape) for shape in shapes]
        gradients_norm_sum = [K.zeros(shape) for shape in shapes]
for p, g, li, ri, twi, gsi, gns in zip(params, grads, L, reward,
                                             tilde_w,gradients_sum,
                                               gradients_norm_sum):
            grad_sum_update = gsi + g
            grad_norm_sum_update = gns + K.abs(g)
            l_update = K.maximum(li, K.abs(g))
            reward_update = K.maximum(ri - g * twi, 0)
            new_w = - grad_sum_update / (l_update * (K.maximum(grad_norm_sum_update + l_update, self._alpha * l_update))) * (reward_update + l_update)
            param_update = p - twi + new_w
            tilde_w_update = new_w
            self.updates.append(K.update(gsi, grad_sum_update))
            self.updates.append(K.update(gns, grad_norm_sum_update))
            self.updates.append(K.update(li, l_update))
            self.updates.append(K.update(ri, reward_update))
            self.updates.append(K.update(p, param_update))
            self.updates.append(K.update(twi, tilde_w_update))
return self.updates
def get_config(self):
        config = {'alpha': float(K.get_value(self._alpha)) }
        base_config = super(COCOB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
"""