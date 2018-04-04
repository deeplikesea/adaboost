import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection

def Weight(n):
    np.random.seed(789)
    return np.transpose([np.random.randn(n)])

def learn(x, weight, c, eta = 0.01):
    for i in range(len(x)):
        if np.dot(x[i],weight) >= 0:
            h = 1
        else:
            h = 0
        if h != c[i]:
            for j in range(len(weight)):
                weight[j] += eta*(c[i]-h)*x[i][j]
    return weight

def evaluate(x, weight, c):
    error = 0
    mislabel = []
    for i in range(len(x)):
       if np.dot(x[i],weight) >= 0:
           h = 1
       else:
           h = 0
       if h != c[i]:
           error += 1
           mislabel.append(i)
    #print("error rate: {}/{}".format(error,len(x)))
    return float(error/len(x)), mislabel

'''
def covariance(x):
    attribute2 = []
    m = len(x)
    n = len(x[0])
    for i in range(m):
        new = []
        for j in range(n):
            for k in range(j,n):
                new.append(x[i][j]*x[i][k])
        attribute2.append(new)
    polynomial2 = np.array(attribute2)
    return polynomial2
'''
def ab_getprob(mislabel, p):
    if mislabel == []:
        return p
    else:
        eps = sum([p[i] for i in mislabel])
        beta = eps/(1-eps)
        for index in range(len(p)):
            if index not in mislabel:
                p[index] = p[index]*beta
        norm = 1/sum(p)
        p = np.array(p)*norm
    return p

def ab_getdata(data,p,pencentage = 0.8):
    subdata = []
    n = len(data)
    index = np.arange(n)
    subn = int(pencentage*float(n))
    for i in range(subn):
        flag  = 0
        while flag == 0:
            s = np.random.choice(index,p = p)
            if s not in subdata:
                subdata.append(s)
                flag = 1
    sub = []
    for i in subdata:
        sub.append(data[i,:])
    return np.array(sub)

def ab_h(x, weight, c):
    error = 0
    h = []
    for i in range(len(x)):
       if np.dot(x[i],weight) >= 0:
           h.append(1)
       else:
           h.append(0)
       if h[i] != c[i]:
           error += 1
    #print("error rate: {}/{}".format(error,len(x)))
    return float((len(x)-error)/len(x)), h 

def ab_voting(h,acc,c):
    error = 0
    for i in range(len(c)):
        for j in range(len(h)):
            if h[j,i] == 0:
                h[j,i] = -1
        h_w = sum([h[x,i]*acc[x] for x in range(len(acc))])
        if h_w >= 0 and c[i] == 0:
            error += 1
        elif h_w < 0 and c[i] == 1:
            error += 1
    return float(error/len(c))

data = np.loadtxt(open('(add a path here)/banknote.csv'),delimiter=",",skiprows=0) #convert cvs table into numberical
np.random.shuffle(data)#shuffle data
#deal with data
linearx = preprocessing.normalize(data[:,:-1])#attributes
c = np.array([[y[-1]] for y in data])#label
data1 = np.concatenate((linearx,c), axis=1)
w0 = np.array([[1] for i in range(len(data))])#w0
data2 = np.concatenate((w0,data1), axis=1)
print("good")
train, test = model_selection.train_test_split(data2,test_size = 0.2)
# deal with our train data to get initial prob list
n = len(train)
prob = 1/float(n)
p = [prob]*n


y_ab = []
y_ab_train = []
y_non = []
step = 15
for num in range(step):
    ###############you can change num to get the # of classifiers you want
    i = 0
    #########for voting
    h = []
    acc = []
    h_train = []
    acc_train = []
    #########
    w = Weight(len(train[0,:-1]))
    while i < num:
        subdata = ab_getdata(train,p)
        w = learn(subdata[:,:-1],w,subdata[:,-1])
        err_rate,mislabel =  evaluate(train[:,:-1],w,train[:,-1])
        p = ab_getprob(mislabel, p)
        acc.append(ab_h(test[:,:-1],w,test[:,-1])[0])
        h.append(ab_h(test[:,:-1],w,test[:,-1])[1])
        acc_train.append(ab_h(train[:,:-1],w,train[:,-1])[0])
        h_train.append(ab_h(train[:,:-1],w,train[:,-1])[1])
        i +=1   
    y_ab.append(ab_voting(np.array(h),np.array(acc),test[:,-1]))
    y_ab_train.append(ab_voting(np.array(h_train),np.array(acc_train),train[:,-1]))
    print("1")
x = [x+2 for x in range(step)]
plt.figure(figsize=(8,4)) #创建绘图对象  
plt.plot(x,y_ab,"b--",linewidth=1)
plt.plot(x,y_ab_train,"r--",linewidth=1)#在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）  
plt.xlabel("# of iteration") #X轴标签  
plt.ylabel("error rate")  #Y轴标签  
plt.title("blue:adaboost") #图标题  
plt.show()  #显示图  
#plt.savefig("line.jpg") #保存图  
