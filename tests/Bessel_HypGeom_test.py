from scipy.special import jv, hyp0f1
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def integrand_bessel(x):
    return jv(1, x)

def integrand_hypergeom(x):
    return (x/2) * hyp0f1(2.0, -x**2 / 4)

# Plot the functions as functions of x to visually confirm they match the expected behavior.
x_values = np.linspace(1000, 10000, 1000)
bessel_values = jv(1, x_values)
hypergeom_values = (x_values / 2) * hyp0f1(2.0, -x_values**2 / 4)   
plt.plot(x_values, bessel_values, label='J1(x)')
plt.plot(x_values, hypergeom_values, label='(x/2) * hyp0f1(2.0, -x^2/4)')
plt.title('Comparison of Bessel Function J1(x) and Hypergeometric Representation')
plt.xlabel('x')
plt.ylabel('Function Value')
plt.grid()
plt.legend()
plt.show()


# Numerically integrate both the Bessel function and the hypergeometric function to verify that they match when multiplied by the appropriate factors.
bessel_integral, _ = quad(integrand_bessel, 0, 10)
hypergeom_integral, _ = quad(integrand_hypergeom, 0, 10)