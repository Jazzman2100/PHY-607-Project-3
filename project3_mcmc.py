import numpy as np
import matplotlib.pyplot as plt 
import tqdm

'''
Data, Signal and Noise Setup
'''
# init. var.

# signal param (actual signal sent)
a = 4
b = np.pi
c = 0.5*np.pi

# model param (our prediction)
A = 6
B = np.pi
C = 0.5*np.pi

# data, signal, model (measured data)
x = np.linspace(0,0.5*np.pi,1000)
signal = a*np.sin(b*x + c)
model = A*np.sin(B*x + C)
data = signal + np.random.normal(0,3,1000)

# plot of data and signal
plt.figure()
plt.title("Data vs Signal")
plt.plot(x,signal, color="yellow", label="signal")
plt.plot(x,model, color="red", label="model")
plt.scatter(x,data, label="data")
plt.legend()

'''
Posterior probability to be sampled from
'''
x = np.linspace(0,0.5*np.pi,1000)

def signal(x):
    return a*np.sin(b*x + c)
def model(x):
     return A*np.sin(B*x + C)
def data(x):
    return signal(x) + np.random.normal(0,3,1000)

# prior
prior = 10/len(x)

# log_likelihood
def post(x):
    return np.sum(-0.5*np.square(data(x)-model(x))) + np.log(prior)

# proposal function
def prop(x):
    return (1 / (2 * np.pi) ** 0.5) * np.exp(-0.5 * x ** 2)

def mcmc(initial, post, prop, iterations):
    x = [initial]
    p = [post(x[-1])]
    for i in tqdm.tqdm(range(iterations)):
        x_test = prop(x[-1])
        p_test = post(x_test)

        acc = p_test / p[-1]
        u = np.random.uniform(0, 1)
        if u <= acc:
            x.append(x_test)
            p.append(p_test)
    return x, p

chain, prob = mcmc(prior, post, prop, 10000)

plt.figure()
plt.title("Evolution of the walker")
plt.plot(chain)
plt.ylabel('x-value')
plt.xlabel('Iteration')

plt.figure()
plt.title("Posterior samples")
plt.hist(chain[100::100], bins=100)
