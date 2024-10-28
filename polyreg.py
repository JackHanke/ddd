import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
num_samps = 15
rand_samples = [(2*i)-1 for i in np.random.rand(1,num_samps)]
def true_f(x): return 2*x + np.cos(25*x)
rand_ys = [true_f(i) for i in rand_samples]


print(rand_samples)
print(rand_ys)


def poly_reg(x_data, y_data, degree, loops, learing_rate):
    weights = np.array([0 for _ in range(0,degree+1)])
    variables = lambda x: np.array([(x**i) for i in range(0,degree+1)])
    model = np.dot(variables, weights.transpose())

    for _ in range(loops):
        wegihts -= learning_rate

    return weights, variables

def test_model(model, given_xs, true_ys): 
    delta = model(given_xs)
    return np.dot(delta, delta.dot())/(delta.shape[0])

for deg in range(0,25):
    model = poly_reg(x_data=rand_samples, y_data=rand_ys, degree=deg, loops=1000, learing_rate=0.01)

    given_xs = [(2*i)-1 for i in np.random.rand(1,2000)]
    mse = test_model(model=model, given_xs=given_xs, true_ys=[true_f(x) for i in given_xs])

    print(f'Best MSE for model of degree {deg} = {mse}')



