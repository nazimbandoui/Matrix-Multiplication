
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import time


def is2k(a):
    n=np.log(a)/np.log(2)
    x=int(n)
    if x==n:
        return -1
    else:
        return x+1

#1ere Mutiplication
def mul1(a,b):
    m=len(a)
    p=len(a[0])
    n=len(b[0])
    result=np.zeros((m,n),dtype=int)
    for i in range(0,m):
        for j in range(0,n):
            for k in range(0,p):
                result[i][j]=result[i][j]+a[i][k]*b[k][j]
    return result

#2eme multiplication
def mul2(a,b):
    if(len(a)<=2 and len(a[0])<=2):
        return np.matmul(a,b)
    wa,xa,ya,za=div4(a)
    wb,xb,yb,zb=div4(b)
    return (recover4(mul2(wa,wb)+mul2(xa,yb),mul2(wa,xb)+mul2(xa,zb),mul2(ya,wb)+mul2(za,yb),mul2(ya,xb)+mul2(za,zb))) 


def arraytolist(x):
    y=[]
    for i in x:
        y.append(i)
    return y


def mul3(aa,bb):
    if(len(aa)<=2 and len(aa[0])<=2):
        return np.matmul(aa,bb)
    a,b,c,d=div4(aa)
    e,f,g,h=div4(bb) 

    return recover4(mul3(b-d,g+h)+mul3(a+d,e+h)+mul3(d,g-e)-mul3(a+b,h),mul3(a,f-h)+mul3(a+b,h),mul3(c+d,e)+mul3(d,g-e),mul3(a,f-h)+mul3(a+d,e+h)-mul3(c+d,e)-mul3(a-c,e+f))


       


#Te9ssem matrice 3la 4
def div4(a):
    w=x=y=z=[]
    f=int(len(a)/2)
    d=len(a)
    return submatrix(a,0,f,0,f) ,submatrix(a,0,f,f,d) ,submatrix(a,f,d,0,f),submatrix(a,f,d,f,d)

#Tfusionner 4 matrices
def submatrix(m,a,b,x,y):
    return (m[a:b:1,x:y:1])

def recover4(a,b,c,d):
    x=np.concatenate((a,b),axis=1)
    y=np.concatenate((c,d),axis=1)
    z=np.concatenate((x,y),axis=0)
    return z

#tekhdem matrice aleatoire de taille n,m
def buildmatrix(size):
    if is2k(size)>=0:
        n=2**is2k(size)
    else :
        n=size
    f=[]
    for i in range(0,n):
        t=[]
        if i>size-1:
            f.append(n0(n))
        else:
            for j in range(0,n) :
                if j>size-1:
                    t.append(0)
                else:
                    t.append(np.random.randint(0,high=10))
            f.append(np.array(t))
    return np.array(f)

def n0(a):
    x=[]
    for i in range(0,a):
        x.append(0)
    return x

#Sous matrice men m men indice a heta b f es lignes et c heta d fles colonnes

###################################################################################################
##################################################################################################
def calculatetime(a,b,c):
    if c==1:
        ts=time.process_time()
        d=mul1(a,b)
        return((time.process_time()-ts)*1000)
    if c==2:
        ts=time.process_time()
        d=mul2(a,b)
        return((time.process_time()-ts)*1000)
    if c==3:
        ts=time.process_time()
        d=mul3(a,b)
        return((time.process_time()-ts)*1000)

def execution(w,x,y,z,g):
    d=[]
    r=[]
    if y==1:
        for i in range(0,g):
            s=0
            for data in range (0,len(w)):
                s+=(calculatetime(w[data],x[data],y))
            r.append(s)
        z.append(np.mean(np.array(r)))
    if y==2:
        for i in range(0,g):
            s=0
            for data in range (0,len(w)):
                s+=(calculatetime(w[data],x[data],y))
            r.append(s)
        z.append(np.mean(np.array(r)))
    if y==3:
        for i in range(0,g):
            s=0
            for data in range (0,len(w)):
                s+=(calculatetime(w[data],x[data],y))
            r.append(s)
        z.append(np.mean(np.array(r)))

	

def createnmatrix(n,size):
    x=[]
    for i in range (0,n):
        x.append(buildmatrix(size))
    return x

def databuilder(datsize,ms):
    c=[]
    for i in datsize:
        c.append(createnmatrix(i,ms))
    return c

def inter(step,times):
    x=[0]
    for i in range(1,times+1):
        x.append(step*i)
    print(x)
    return x



datasize=inter(100,20)
#datasize=[1,2,10,25,50,100,200,500,1000,1500,2000,4000,3000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000]
#datasize=[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
executiontime1=[]
executiontime2=[]
executiontime3=[]
nb=8

datax=databuilder(datasize,nb)
datay=databuilder(datasize,nb)
times=1
for i in range(0,len(datasize)):
	execution(datax[i],datay[i],1,executiontime1,times)
	#execution(datax[i],datay[i],2,executiontime2,times)
	#execution(datax[i],datay[i],3,executiontime3,times)

print(executiontime1)
#plt.plot(datasize,executiontime1,datasize,executiontime2,datasize,executiontime3)
#plt.legend(labels=["Classic Multiplication","Divide & Conquer","Strassen's Algorithm"])
plt.plot(datasize,executiontime1,'r')
plt.legend(labels=["Classic Multiplication"])
plt.xlabel("Matrix number")
plt.ylabel("Time (ms)")
plt.show()
