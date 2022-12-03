import scipy.special
from scipy.special import lambertw, factorial, binom
import numpy as np
import matplotlib.pyplot as plt
from mpmath import nsum

def qWinv(a, x):
    return x*np.exp(a*(x**2)+x)

def qW(a, x):
    num_terms = 1000
    #qW = 0
    print(x)
    print(np.shape(x))
    qW = np.zeros((num_terms, np.size(x,0)))
    S = 0
    for k in np.arange(1,num_terms):
        s = 0
        for m in np.arange(np.floor((k-1)/2), k):
            s = s \
                + ((-k)**(m) / (a)**(m+1))\
                /(factorial(m))\
                * binom(m, k-1-m)
            print(s)
        S = S + ((a*x)**(k)/(k) * s)
        qW[k,:] = S
    return qW

def qW_mp(a, x):
    qW = 0
    qW = nsum(lambda k, m: k+m, [2,3], [4,5])
    return qW

def qW_newton(a, x):
    x0 = np.zeros_like(x)
    n_iters = 10

    S = x0
    for k in np.arange(0, n_iters):
        S = S - (qWinv(a,S) - x)/((2*a*S**2 + S + 1)*np.exp(a*S**2 + S))
    return S

def g_fun(a, x, z):
    return x*np.exp(a*x**2 + x) - z

def g_p_fun(a, x):
    return (2*a*x**2 + x + 1)*np.exp(a*x**2+x)

def g_pp_fun(a, x):
    return (4*a**2 * x**3 + 4*a*x**2 + (6*a+1)*x + 2)*np.exp(a*x**2 + x)

def gpp_gp_fun(a, x):
    return (4*a*x+1)/(2*a*x**2+x+1) + 2*a*x + 1

def gp_g_fun(a, x, z):
    return x/(2*a*x**2+x+1) - z*(2*a*x**2+x+1)*np.exp(-a*x**2 - x)

def qW_halley(a, z, n_iters):
	x0 = np.zeros_like(z)
	#n_iters=18
	S = x0
	for k in np.arange(0, n_iters):
		g = g_fun(a,S,z)
		g_p = g_p_fun(a,S)
		g_pp = g_pp_fun(a,S)
		S = S - (2*g*g_p)/(2*g_p**2 - g*g_pp)

		#gp_g = gp_g_fun(a, S, z)
		#gpp_gp = gpp_gp_fun(a, S)
		#S = S - gp_g / (1 - gp_g*gpp_gp)
	return S

a = -1
res = 1000

alpha_a = (-1 + np.sqrt(1-8*a))/(4*a)
beta_a = (-1 - np.sqrt(1-8*a))/(4*a)

x_min = qWinv(a, alpha_a)
x_max = qWinv(a, beta_a)

print("alpha_a: ", alpha_a)
print("beta_a: ", beta_a)
print("f(alpha_a): ", x_min)
print("f(beta_a): ", x_max)

xs = np.linspace(x_min, x_max, res)

y_h_1 = qW_halley(a, xs, 18)
y_h_2 = qW_halley(a, xs, 20)

plt.plot(xs, y_h_1)
plt.plot(xs, y_h_2)
plt.axis('equal')
plt.figure(2)
plt.plot(xs, np.abs(y_h_2-y_h_1))
plt.show()
