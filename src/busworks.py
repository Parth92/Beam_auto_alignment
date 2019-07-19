from pymodbus.client.sync import ModbusTcpClient
import numpy as np
from time import time
from time import sleep
from copy import copy
#import asyncio
from warnings import warn

class BaseException(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg
    

class BusWorks_DAC(object):
    def __init__(self, address='192.168.1.231', port=502, num_chns=8):
        self.address = address
        self.port = port
        self.p = np.array([3000.21057623, -16.13130127])
        self.dt = 0.01
        self.num_chns = num_chns
    @property
    def dt(self):
        return self._dt
    
    @dt.setter
    def dt(self, value):
        if value < 3e-3:
            warn("The update time dt of ACROMAG 1541-000 is often greater than {} ms. Default dt = 10 ms.".format(value*1e3))
        elif value < 5e-3:
            warn("The update time dt of ACROMAG 1541-000 is occassionally greater than {} ms. Default dt = 10 ms.".format(value*1e3))
        elif value < 1e-2:
            warn("The update time dt of ACROMAG 1541-000 is rarely greater than {} ms, but it happens. Default dt = 10 ms.".format(value*1e3))
        self._dt = value
            
    @property
    def address(self):
        return self._address
    @address.setter
    def address(self, value):
        if not isinstance(value, str):
            raise BaseException('Address must be a string. Input is of type {}'.format(type(value)))
        self._address = value
        
    @property
    def port(self):
        return self._port
    @port.setter
    def port(self, value):
        if not isinstance(value, int):
            raise BaseException('Port must be an int. Input is of type {}'.format(type(value)))
        self._port = value
        
    def start(self):
        self.DAC = ModbusTcpClient(self.address, port=self.port)
        
    def stop(self, reset=True):
        self.DAC.write_registers(1, [0]*self.num_chns)
        self.DAC.close()
        
    def set_voltages(self, Vs, start_ch):
        if not isinstance(Vs, np.ndarray) and not isinstance(Vs, list) and not isinstance(Vs, tuple):
            raise BaseException('Vs must be a numpy.array, list, or tuple. Not {}'.format(type(Vs)))
        n = list(self.V2int(np.array(Vs)))
        t0 = time()
        self.DAC.write_registers(start_ch, n)
        sleep(max(0, self.dt - time() + t0))

    def set_voltage(self, V, ch=1):
        if not isinstance(V, int) and not isinstance(V, float):
            raise BaseException('Voltage must be a number. Input is of type {}'.format(type(V)))
        n = self.V2int(V)
        t0 = time()
        self.DAC.write_register(ch, n)
        sleep(max(0, self.dt - time() + t0))
        
    def loop_sinuses(self, A, f, T, phi=0, start_ch=1):
        
        # Checking input types
        if isinstance(A, np.ndarray) or isinstance(A,list) or isinstance(A, tuple):
            A = np.array(A).astype(float)
        elif isinstance(A, int) or isinstance(A, float):
            A = np.array([A]).astype(float)
        else:
            raise BaseException('A must be iterable or a number, not {}'.format(type(A)))
            
        if isinstance(f, np.ndarray) or isinstance(f,list) or isinstance(f, tuple):
            f = np.array(f).astype(float)
        elif isinstance(f, int) or isinstance(f, float):
            f = np.array([f]).astype(float)
        else:
            raise BaseException('f must be iterable or a number, not {}'.format(type(f)))

        if isinstance(phi, np.ndarray) or isinstance(phi,list) or isinstance(phi, tuple):
            phi = np.array(phi).astype(float)
        elif isinstance(phi, float) or isinstance(phi,int):
            phi = np.array([phi]).astype(float)
        else:
            raise BaseException('phi must be iterable or a number, not {}'.format(type(phi)))
        phi *= np.pi/180.0
        
        dt = self.dt
        # Checking lengths of inputs        
        NA = len(A)
        Nf = len(f)
        Nphi = len(phi)
        Nin = np.array([NA, Nf, Nphi])
        nCh = Nin.max()
        if any(Nin != nCh):
            if len(A) != nCh:
                if len(A) == 1:
                    A = np.ones(nCh)*A
                else:
                    raise BaseException('Inputs A, f and phi must be of the same length, or be scalar')
            if len(f) != nCh:
                if len(f) == 1:
                    f = np.ones(nCh)*f
                else:
                    raise BaseException('Inputs A, f and phi must be of the same length, or be scalar')
            if len(phi) != nCh:
                if len(phi) == 1:
                    phi = np.ones(nCh)*phi
                else:
                    raise BaseException('Inputs A, f and phi must be of the same length, or be scalar')
        
        
        res = (1.0/f)%dt
        tol = 1e-5
        for k,r in enumerate(res):
            if not (r/dt < tol or (dt - r)/dt < tol):
                N1 = np.round(1.0/(f[k]*dt))
                f1 = (1.0/(N1*dt))
                print('Adjusting frequency: {} Hz ---> {:.3f} Hz (1/f = integer number of dt = {} s)'.format(f[k], f1, dt))
                f[k] = f1
        N = np.round(1.0/(f*dt) + 1.0).astype(int)  # Number of data points in one period
        S = []
        for k in range(nCh):     
            t = np.linspace(0, 1.0/f[k], N[k])[:-1] # Samplig time array
            # Array with one period of the wanted sinus function
            S.append(A[k]*np.sin(2.0*np.pi*f[k]*t + phi[k]))

        # Looping the sinus functions
        self.loop_funcs(S, T, start_ch)
       
        
        
    def loop_sinus(self, A, f, T, phi=0, ch=1):
        
        dt = self.dt
        phi *= np.pi/180.0
        N = round(1.0/(f*dt) + 1.0)  # Number of data points in one period
        # Redefining dt to be consistent with an integer number of data points
        dt = 1.0/(f*(N-1.0))
        t = np.linspace(0, 1.0/f, N)[:-1] # Samplig time array
        # New number of data points since we removed the last data point.
        N = len(t)
        # Array with one period of the wanted sinus function
        S = A*np.sin(2.0*np.pi*f*t + phi)
        # Looping the sinus function
        self.loop_func(S, T, ch)
        
        
    def loop_funcs(self, F, T, start_ch=1):
        dt = self.dt
        N = np.zeros(len(F), dtype=int)
        Fn = []
        for k,f in enumerate(F):
            Fn.append(self.V2int(f))
            N[k] = len(f)
        k = 0
        t0 = time()
        t1 = time()
        while t1-t0 < T:
            ls = []
            for i, f in enumerate(Fn):
                ls.append(f[k%N[i]])
            # Making sure that there is dt secs between each DAC-call.
            # 7e-5 is a measured average deviation.
            sleep(max(0, dt-(time()-t1) - 8.4e-5))
            t1 = time()
            self.DAC.write_registers(start_ch, ls )
            k += 1
               
        # Setting the voltage to zero when done.
        self.set_voltages(np.zeros(len(F)), start_ch)
    
    def loop_func(self, F, T, ch=1):
        dt = self.dt
        # Converting to bins instead of Volts. 
        F = self.V2int(F)
        N = len(F)
        k = 0
        t0 = time()
        t1 = time()
        while t1-t0 < T:
            # Making sure that there is dt secs between each DAC-call.
            # 7e-5 is a measured average deviation.
            sleep(max(0, dt-(time()-t1) - 7e-5))
            t1 = time()
            self.DAC.write_register(ch, F[k%N])
            k += 1

        # Setting the voltage to zero when done.
        self.set_voltage(0,ch)

        # Setting voltage back to zero.
        # self.set_voltage(0,ch)
    
    def V2int(self, V):
        if isinstance(V, float) or isinstance(V, int):
            n = 0
            for k,p in enumerate(self.p[::-1]):
                n += p*V**k
            n = int(round(n))
            if n < 0:
                n += 65536
        elif isinstance(V, np.ndarray):
            n = np.zeros(len(V))
            for k,p in enumerate(self.p[::-1]):
                n += p*V**k
            n = np.round(n)
            n[n < 0] += 65536
            n = n.astype(int)
        else:
            raise BaseException('V must be a number or a numpy.array, not {}'.format(type(V)))
        return n
    
    def int2V(self, n):
        if len(self.p) != 2:
            raise BaseException('Currently, int2V() only works for linear transformation. Thus, if len(self.p = 2)')
        if isinstance(n, list) or isinstance(n, tuple) or isinstance(n, np.ndarray):
            n = np.array(n)
            n[n>32767] -= 65536
        elif isinstance(n, int) or isinstance(n, float):
            if n > 32767:
                n -= 65536
        else:
            raise BaseException('Input must be int, float or np.array, not {}'.format(type(n)))
        V = (n-self.p[1])/self.p[0]
        return V
    
        
    def calibrate(self, V, n, order=1, verbose=True):
        V = copy(V)
        n = copy(n)
        tjena = 'heh'
        if (not isinstance(V, np.ndarray) and not isinstance(V, list) and 
            not isinstance(V, tuple)):
            raise BaseException('V (Voltage data) must be a numpy.array, list, or tuple. Not {}'.format(type(V)))
        if (not isinstance(n, np.ndarray) and not isinstance(n, list) and 
            not isinstance(n, tuple)):
            raise BaseException('n (integer data) must be a numpy.array, list, or tuple. Not {}'.format(type(n)))
        if len(V) != len(n):
            raise BaseException('V (Voltage data) and n (integer data) must have the same length {}')
        if len(V) < order+1:
            raise BaseException('len(V) = len(n) = {} <= deg = {}!!'.format(len(V), order)) 

        n[n>32767] -= 65536
        p = np.polyfit(V, n, order)
        self.p = p
        if verbose:
            txt = 'Calibration result: n = '
            for k, c in enumerate(p):
                if c < 0:
                    txt += '-'
                    if k > 0:
                        txt+=' '
                elif k > 0:
                    txt += '+ '
                txt += '{}'.format(abs(c))

                if k < len(p)-1:
                    txt += '*V'.format(len(p)-k-1)
                if k < len(p)-2:
                    txt += '**{} '.format(len(p)-k-1)
                else:
                    txt+=' '
            print(txt)
        
    def read_registers(self, start_ch=1, num_channels=4):
        out = self.DAC.read_holding_registers(start_ch, num_channels)
        return out.registers
    
    def read_voltages(self, start_ch=1, num_channels=4):
        return self.int2V(self.read_registers(start_ch, num_channels))
        
def lin_eq(p1,p2):
    k = (p2[1] - p1[1])/(p2[0] - p1[0])
    m = p1[1] - k*p1[0]
    return k,m
 
def y(x,k,m):
    return k*x + m

def V2int(V, p_max=[10.0, 32767], p_min = [0,0], n_max=[0,65535], n_min=[-10.0, 32768]):
    if V<0:
        k,m = lin_eq(n_min, n_max)
        n = round(y(V,k,m))
    else:
        k,m = lin_eq(p_min, p_max)
        n = round(y(V,k,m))
    #if V>0:
    #    k = (p_int_max - p_int_min)/(p_V_max - p_V_min)
    #    m = p_int_min - k*
    #    int_out = round(V*(p_int_max - p_int_min)/(p_V_max - p_V_min))
    #else:
    #    int_out = round(V*(n_int_max - n_int_min)/(n_V_max - n_V_min))
        
    return n

def pearson(x,y):
    return ((x - x.mean())* (y - y.mean())).sum() / ( np.sqrt( (( x-x.mean() )**2).sum() ) * 
                                                      np.sqrt( (( y-y.mean() )**2).sum() ) )

def max_deviation(x,y,k,m):
    return np.abs(y - (k*x + m)).max()

def test(a,b,c):
    print(vars())
