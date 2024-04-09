# Basic Notions of Probability
:::success
[回隨機微積分](/4m94zjAmQYSZ6dupjdsNRw)
編輯:2023/04/09
此筆記的ipynd檔可以在我的[Github](https://github.com/Iofting1023)找到．
:::
[TOC]
## 1.1. Distributions as histograms.
#### (a) 
Sample $𝑁 = 10,000$ random numbers uniformly on $[0, 1]$.
```python=+
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1023)
#a
N = 10000
sample = np.random.rand(N)
sample
```
```
array([0.59379366, 0.85444082, 0.03805208, ..., 0.85521843, 0.10999541,
       0.67130031])
```
#### (b) 
Plotting the PDF: Divide the interval $[0,1]$ into $𝑚$ bins $𝐵_1,\ldots,𝐵_{50}$ of equal length $1/50$. Plot the histogram of the values for $50$ bins where the value at bin $j$ is $$\frac{\#\{i\leq N:X_i\in B_j\}}{N}$$
```python=+
plt.hist(sample , density =True,bins =50)
plt.show()
```
![image](https://hackmd.io/_uploads/H1UPjjZgA.png)

#### (c)
Plotting the CDF: Plot the cumulative histogram of the values for $50$ bins where the value at each bin $𝑗$ is
$$\frac{\#\{i\leq N:X_i\leq j/50\}}{N}$$
```python=+
#c
plt.hist(sample , density =True,bins =50, cumulative = True)
plt.show()
```
![image](https://hackmd.io/_uploads/BJ3yhsWx0.png)

#### (d)
Re-do items (b) and (c) for the square of each number. This would approximate the PDF of $X^2$ where 𝑋 is uniformly distributed on $[0, 1]$.
```python=+
sample_2 = sample**2
plt.hist(sample_2 , density =True,bins =50)
plt.show()
plt.hist(sample_2 , density =True,bins =50, cumulative = True)
plt.show()
```
![image](https://hackmd.io/_uploads/BJzUhiZlR.png)
![image](https://hackmd.io/_uploads/ByLD3obeA.png)

## 1.2. The law of large numbers.
#### (a)The strong law.
Sample $𝑁 = 10,000$ random numbers $𝑋_1,\ldots, 𝑋_𝑁$ exponentially distributed with parameter 1. Use the command ``numpy.cumsum`` to
get the empirical mean $\frac{𝑆_𝑁}{N} = \frac{1}{N}(𝑋_1+\dots+𝑋_𝑁)$.Plot the values for $𝑁=
1,\dots, 10,000.$ What do you notice?
$$P(\lim_{n\rightarrow \infty}\frac{S_N}{N}=1)=1$$
此指數分配的期望值為1
```python=+
N = 10000
sample = np.random.exponential(1,N)
S_n = [np.cumsum(sample)[i]/(i+1) for i in range(10000)]
plt.plot(S_n)
```
![image](https://hackmd.io/_uploads/Sk0Xl2WeA.png)
#### (b)The weak law.
Define a function in Python using the command ``def`` that returns the empirical mean of a sample of size $𝑁$ as above. This will allow you to sample the empirical mean $\frac{𝑆_𝑁}{N}$ as many times as needed for a given $𝑁$.Plot the histograms (PDF and CDF) of a sample of size 10,000 of the empirical
mean for $𝑁 = 100$ and $𝑁 = 10,000$. What do you notice?
```python=+
def sample_mean(N,times):
    mean = []
    for i in range(times):
        mean.append(np.random.exponential(1,N).mean())
    return mean
t_100 = sample_mean(100,10000)
t_10000 = sample_mean(10000,10000)
plt.hist(t_100, density = True)
plt.hist(t_10000, density = True)
plt.show()
```
![image](https://hackmd.io/_uploads/Hklzz3-l0.png)
與上一小題不同的是(a)小題是直接畫出$\frac{S_N}{N}$隨$N$變大值的變化，這一小題是看給定$N$,$\frac{S_N}{N}$抽樣許多次的值，想描述的是大數弱法則$$\lim_{n\rightarrow \infty}P(\frac{S_N}{N}=1)=1$$
可以預期的是當$N$上升，抽樣出的樣本值離1越來越近，也就是分佈收斂到單點1，同時也表示機率收斂到1.
## 1.3 The central limit theorem
The approximation of the expectation in terms of the empirical mean is not exact for finite $𝑁$. The error is controlled by the central limit theorem. For a sample $𝑋_1 , \ldots , 𝑋_𝑁$ of the random variable $𝑋$ of mean $E[𝑋 ]$ and variance $\sigma^2$, this theorem says that the sum $𝑆_𝑁 = 𝑋_1 +\ldots+ 𝑋_𝑁$ behaves like $𝑆_𝑁 \approx 𝑁E[𝑋]+\sqrt{𝑁}\sigma𝑍$, where $𝑍$ is a standard Gaussian random variable. More precisely, this means that
$$\lim_{n\rightarrow \infty}\frac{S_N-NE[X]}{\sigma\sqrt{N}}=Z$$
The limit should be understood here as the convergence in distribution. Practically speaking, this means that the histogram of the random variable $\frac{𝑆_𝑁−𝑁E[𝑋]}{\sigma\sqrt{N}}$ should resemble the one of a standard Gaussian variable when $𝑁$ is large.We check this numerically.
#### (a)
Let $𝑆_𝑁 = 𝑋_1 +\ldots + 𝑋_𝑁$ where the $𝑋_𝑖$ are exponentially distributed random variables of parameter $1$. Define a function in Python using the command ``def`` that returns for a given $𝑁$ the value $𝑌_𝑁 =\frac{𝑆_𝑁−𝑁E[𝑋]}{\sigma\sqrt{N}}$
```python=+
def y_n(N):
    sample = np.random.exponential(1,N)
    standard = (sample.sum()-N*1)/1
    return standard
```

#### (b)
Plot the histograms (PDF) of a sample of size 10,000 of $𝑌_𝑁$ for $𝑁 = 100$. What do you notice?
```python=+
#b
N = 100
size = 10000
sample = [y_n(N) for i in range(size)]
plt.hist(sample , density =True,bins =50)
plt.show()
```
![image](https://hackmd.io/_uploads/SktLdUGe0.png)
乍看跟常態差不多
#### (c)
Compare the above to the histogram of a sample of size $10,000$ of points generated using the standard Gaussian distribution.
```python=+
sample = np.random.normal(0,1,10000)
plt.hist(sample, density =True,bins =50)
plt.show()
```
![image](https://hackmd.io/_uploads/SkJyK8fx0.png)
跟b小題形狀很像．
## 1.4 Sampling Cauchy random variables.
shows that there is no law of large numbers (weak or strong) for Cauchy random variables.
#### (a)
Let $𝐹_X^{−1}$ be the inverse of the CDF of the Cauchy distribution. Plot the histogram of $𝐹_X^{−1}(𝑈)$ where $𝑈$ is a uniform random variable for $𝑋$ sample of $10,000$ points. Use $100$ bins in the interval $[-10,10]$.
- First compute $F^{-1}_X$, since $F_X(x)=\frac{1}{\pi}\tan^{-1}x+\frac{1}{2}$, we can find $F^{-1}_X(x)=\tan(\pi(x-\frac{1}{2}))$
```python=+
N = 10000
uni_sample = np.random.rand(N)
cauchy_sample = np.tan(np.pi*(uni_sample - 0.5))
plt.hist(cauchy_sample, density =True,bins =100,range=[-10,10])
plt.show()
```
![image](https://hackmd.io/_uploads/Hkfyn8GxA.png)
#### (b)
Compare the above to the histogram of a sample of size $10,000$ of points generated using the standard Gaussian distribution.
```python=+
#b
sample = np.random.normal(0,1,N)
plt.hist(sample, density =True,bins =100,range=[-10,10])
plt.show()
```
![image](https://hackmd.io/_uploads/HkA72IGlC.png)
可以明顯看出從常態抽樣尾端事件的樣本少很多，幾乎是沒有．
#### (c)
 Let ($𝐶_𝑛,𝑛\leq 10,000$) be the values obtained in (a). Plot the empirical mean $\frac{𝑆_𝑁}{N} = \frac{1}{N} \sum_{n\leq N} 𝐶_𝑛$ for $𝑁 = 1,\ldots,10,000$. What do you notice?
```python=+
#c
S_n = [np.cumsum(cauchy_sample)[i]/(i+1) for i in range(10000)]
plt.plot(S_n)
```
![image](https://hackmd.io/_uploads/B1hbCIzeA.png)
樣本平均並沒有隨著$N$(抽樣數變大)有收斂到定值的趨勢．
#### (d)
Define a function in Python using the command ``def`` that returns the empirical mean $\frac{𝑆_𝑁}{N}$ of a sample of size $𝑁$ as above. Plot the histograms (PDF) of a $𝑁$ sample of size $10,000$ of the empirical mean for $𝑁 = 10$ and $𝑁 = 100$. What do you notice compared to the histograms in Project 1.2?
```python=+
size = 10000
def cauchy_sample_mean(N,size):
    mean = []
    for i in range(size):
        uni_sample = np.random.rand(N)
        cauchy_mean = np.tan(np.pi*(uni_sample - 0.5)).mean()
        mean.append(cauchy_mean)
    return mean
plt.hist(cauchy_sample_mean(10,size) , density =True,bins =50,range=[-10,10])
plt.show()
plt.hist(cauchy_sample_mean(100,size) , density =True,bins =50,range=[-10,10])
plt.show()
```
![image](https://hackmd.io/_uploads/ByNNgwGlR.png)
![image](https://hackmd.io/_uploads/S1gSevfl0.png)
可以看出樣本平均的分配仍類似柯西分配，事實上這個性質是成立的．

