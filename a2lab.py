import numpy
import matplotlib.pyplot as plt

# constants
kErg = 1.38065e-16  # erg/K
kEv = 8.61734e-5  # eV/K
h = 6.62607e-27 # erg s
elmass = 9.109390e-28 # electrong mass g



# Compute U_r
chiion = numpy.array([7, 16, 31, 51])  # eV
temps = numpy.array([5000, 10000, 20000])

u = []
for temp in temps:
    for r in range(4):
        U = 0
        for s in range(chiion[r]):
            U += numpy.exp(-1*s/(kEv*temp))
        print("U(r=%i, T=%i) = %.2f"%(r+1, temp, U))
        u.append(U)


def partfunc_E(temp):
    """partition functions Schadee element
    input: temp(K)
    output: numpy array(4), partition functions U1,...U4
    """
    u = []
    for r in range(4):
        U = 0
        for s in range(chiion[r]):
            U += numpy.exp(-1*s/(kEv*temp))
        u.append(U)
    return u


for temp in temps:
    output = partfunc_E(temp)
    # convert to output string
    outStr = " ".join(["%.5f"%x for x in output])
    print("parfunc_E(temp=%i) = %s" %(temp, outStr))


def boltz_E(temp, r, s):
    """compute Boltzmann population for level r, s of Schadee element E
    input: temp (K)
           r, ionization stage nr, 1-4 where 1= neutral E
           s, level nr, starting at s=1
    output: reletive level population n_(r,s)/N-r
    """
    u = partfunc_E(temp)
    relnrs = 1 / u[r-1] * numpy.exp(-(s-1)/(kEv*temp))
    return relnrs

print("")
for s in range(1,11):
    print("%.5e"%boltz_E(5000,1,s))


def saha_E(temp, elpress, ionstage):
    """compute Saha population fraction N_r/N for Schadee element E
    input: temperature (K)
           electron pressure
           ion stage
    return population fraction
    """
    kevT = kEv*temp
    kergT = kErg*temp
    eldens = elpress/kergT
    u = partfunc_E(temp)
    u = u + [2.0]
    # print("uuu", u)
    sahaconst = (2 * numpy.pi * elmass * kergT / (h**2))**1.5 * 2 / eldens
    nstage = numpy.zeros(5)
    nstage[0] = 1
    for r in range(4):
        nstage[r+1] = nstage[r] * sahaconst*u[r+1]/u[r] * numpy.exp(-chiion[r]/kevT)
    ntotal = numpy.sum(nstage)
    nstagerel=nstage/ntotal
    return nstagerel[ionstage-1]

print("")
for r in range(1,6):
    print("%.5e"%(saha_E(20000,1e3,r)))


print("")
for r in range(1,6):
    print("%.5e"%(saha_E(20000,1e1,r)))


def sahabolt_E(temp, elpress, ion, level):
    """compute Saha-Boltzmann population n_(r,s)/N for level r,s of E
    input: temp (K)
           electron pressure
           ionization stage
           level nr
    """
    return saha_E(temp, elpress, ion) * boltz_E(temp, ion, level)


print("")
for s in range(1,6):
    print(sahabolt_E(5000,1e3,1,s))

print("")
for s in range(1,6):
    print(sahabolt_E(20000,1e3,1,s))

print("")
for s in range(1,6):
    print(sahabolt_E(10000,1e3,2,s))

print("")
for s in range(1,6):
    print(sahabolt_E(20000,1e3,4,s))


# plot
temp = numpy.arange(0, 30001, 1000)
pop = numpy.zeros((5, len(temp), 4))
for s in [1,2,3,4]:
    for T in range(1,31):
        for r in range(1,5):
            pop[r,T,s-1] = sahabolt_E(temp[T], 131, r, s)
fig = plt.figure()
for ionStage in [1,2,3,4]:
    for s in [1]:
        plt.semilogy(temp, pop[ionStage, :, s-1], label="ionstage=%i, s=%i"%(ionStage, s))
plt.legend()
plt.title("s=1")
plt.ylim([1e-3, 1.1])
plt.xlabel("temperature")
plt.ylabel("population")
# plt.show()

fig = plt.figure()
for ionStage in [1,2,3,4]:
    for s in [1,2,3,4]:
        plt.semilogy(temp, pop[ionStage, :, s-1], label="ionstage=%i, s=%i"%(ionStage, s))
plt.legend()
plt.title("s=1-4")
plt.ylim([1e-3, 1.1])
plt.xlabel("temperature")
plt.ylabel("population")
# plt.show()

# hydrogen
def sahabolt_H(temp, elpress, level, doPrint=False):
    """compute Saha-Boltzmann population n_(1,s)/N_H for hydrogen level
    intpu: temp (K)
           electron pressure
           level number
    """
    kevT = kEv*temp
    kergT = kErg*temp
    eldens = elpress/kergT
    nrlevels = 100
    g = numpy.zeros((2,nrlevels))
    chiexc = numpy.zeros((2,nrlevels))
    for s in range(nrlevels):
        g[0,s] = 2*(s+1)**2
        chiexc[0,s] = 13.598*(1 - 1/(s+1)**2)
    g[1,0] = 1
    chiexc[1,0] = 0
    u = numpy.zeros(2)
    u[0] = 0
    for s in range(nrlevels):
        u[0] = u[0] + g[0,s] * numpy.exp(-chiexc[0,s]/kevT)
    u[1] = g[1,0]
    sahaconst = (2*numpy.pi*elmass*kergT/h**2)**1.5 * 2/eldens
    nstage = numpy.zeros(2)
    nstage[0] = 1
    nstage[1] = nstage[0] * sahaconst * u[1]/u[0] * numpy.exp(-13.598/kevT)
    ntotal = numpy.sum(nstage)
    nlevel = nstage[0]*g[0, level-1]/u[0]*numpy.exp(-chiexc[0, level-1]/kevT)
    nlevelrel=nlevel/ntotal
    # print parameters
    if doPrint:
        print(u)
        for s in range(6):
            print(s+1, g[0,s], chiexc[0,s], g[0,s]*numpy.exp(-chiexc[0,s]/kevT))
        print("")
        for s in numpy.arange(0,91,10):
            print(s+1, g[0,s], chiexc[0,s], g[0,s]*numpy.exp(-chiexc[0,s]/kevT))
    return nlevelrel

# 2.8
print("")
sahabolt_H(5000,1e2,1, doPrint=True)

def partfunc_Ca(temp):
    """partition functions Ca
    input: temp(K)
    output: numpy array(4), partition functions U1,...U4
    """
    chiion = [6.113, 11.871, 50.91, 67.15]
    u = []
    for r in range(4):
        U = 0
        for s in range(int(numpy.floor(chiion[r]))):
            U += numpy.exp(-1*s/(kEv*temp))
        u.append(U)
    return u

def boltz_Ca(temp, r, s):
    """compute Boltzmann population for level r, s of Schadee element E
    input: temp (K)
           r, ionization stage nr, 1-4 where 1= neutral E
           s, level nr, starting at s=1
    output: reletive level population n_(r,s)/N-r
    """
    u = partfunc_Ca(temp)
    relnrs = 1 / u[r-1] * numpy.exp(-(s-1)/(kEv*temp))
    return relnrs

def saha_Ca(temp, elpress, ionstage):
    """compute Saha population fraction N_r/N for Ca
    input: temperature (K)
           electron pressure
           ion stage
    return population fraction
    """
    chiion = [6.113, 11.871, 50.91, 67.15]
    kevT = kEv*temp
    kergT = kErg*temp
    eldens = elpress/kergT
    u = partfunc_Ca(temp)
    u = u + [2.0]
    # print("uuu", u)
    sahaconst = (2 * numpy.pi * elmass * kergT / (h**2))**1.5 * 2 / eldens
    nstage = numpy.zeros(5)
    nstage[0] = 1
    for r in range(4):
        nstage[r+1] = nstage[r] * sahaconst*u[r+1]/u[r] * numpy.exp(-chiion[r]/kevT)
    ntotal = numpy.sum(nstage)
    nstagerel=nstage/ntotal
    return nstagerel[ionstage-1]

def sahabolt_Ca(temp, elpress, ion, level):
    """compute Saha-Boltzmann population n_(r,s)/N for level r,s of E
    input: temp (K)
           electron pressure
           ionization stage
           level nr
    """
    return saha_Ca(temp, elpress, ion) * boltz_Ca(temp, ion, level)

# page 23 prints
temp = numpy.arange(1000,20001,100)
print("len temp", len(temp))
CaH = numpy.zeros(temp.shape)
Caabund=2e-6
for i in range(191):
    NCa = sahabolt_Ca(temp[i], 1e2, 2, 1)
    NH = sahabolt_H(temp[i], 1e2, 2)
    CaH[i] = NCa*Caabund/NH

plt.figure()
plt.semilogy(temp, CaH)
plt.xlabel("temperature")
plt.ylabel("Ca II K / H alpha")

print("")
print("Ca/H ratio at 5000 K =", CaH[temp==5000])

temp = numpy.arange(2000, 12001, 100)
dNCadT = numpy.zeros(temp.shape)
dNHdT = numpy.zeros(temp.shape)
dT = 1
for i in range(101):
    NCa = sahabolt_Ca(temp[i], 1e2, 2, 1)
    NCa2 = sahabolt_Ca(temp[i]-dT, 1e2, 2, 1)
    dNCadT[i] = (NCa - NCa2)/dT/NCa
    NH = sahabolt_H(temp[i], 1e2, 2)
    NH2 = sahabolt_H(temp[i]-dT, 1e2, 2)
    dNHdT[i] = (NH - NH2)/dT/NH

plt.figure()
plt.semilogy(temp, numpy.abs(dNHdT), label="dNHdT")
plt.ylim([1e-5, 1])
plt.xlabel("temperature")
plt.ylabel("abs d n(r,s) / n(r,s)")
plt.semilogy(temp, numpy.abs(dNCadT), "--", label="dNCadT")
plt.legend()

# 2.10
print("")
for temp in numpy.arange(2000, 20001, 2000):
    print(temp, sahabolt_H(temp, 1e2, 1))

temps = numpy.arange(1000, 20001, 1000)
nH = []
for temp in temps:
    nH.append(sahabolt_H(temp, 1e2, 1))
plt.figure()
plt.plot(temps, nH)
plt.xlabel("temperature")
plt.ylabel("neutral hydrogen fraction")

plt.show()
