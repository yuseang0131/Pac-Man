import math

n = 200
r = (3.1 + 1.9)/2 /2/ 100
dB = - 0.049
dt = 0.940 - 0.817

mean = 0.0349

emf = -n*math.pi*r**2*dB/dt

print(emf, '|', (mean - emf)/emf * 100)
