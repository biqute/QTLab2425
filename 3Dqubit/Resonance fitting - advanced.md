# Resonance fitting

## Theoretical models
Our goal is the characterization of a 3D-superconducting aluminum cavity. We want to determine the quality factor of the cavity as an electromagnetic microwave resonator. To do so, we probe the cavity using a VNA and collect the scattering parameters $S_{11}$ and $S_{21}$ over a range of microwave frequencies. By means of fitting, we extract from this data the internal quality factor $Q_i$ of the cavity, and other useful parameters. The theoretical model used for the scattering parameters are taken from _Characterization of superconducting resonant RF cavities for axion search with the QUAX experiment_ by Alessio Rettaroli, and corrected with an attenuation $A$ and phase differences $\theta_{21}$ and $\theta_{11}$. Defining ${\delta(f, f_0) = f/f_0 - f_0/f}$, we rearrange $S_{11}$ and $S_{21}$ of equations $(2.54)$ and $(2.57)$ as
$$
S_{11}(f) = Ae^{j\theta_{11}} \frac{B(k_1, k_2) + jQ_L\delta(f, f_0)}{1 + jQ_L\delta(f, f_0)}, \tag{1}
$$
$$
S_{21}(f) = Ae^{j\theta_{21}}\frac{C(k_1, k_2)}{1 + jQ_L\delta(f, f_0)}, \tag{2}
$$
with
$$
B(k_1, k_2) = \frac{k_1 - k_2 - 1}{k_1 + k_2 + 1}, \tag{3}
$$
$$
C(k_1, k_2) = \frac{2\sqrt{k_1k_2}}{1 + k_1 + k_2}. \tag{4}
$$
The internal quality factor is given by $Q_i = (1+k_1+k_2)Q_L$. 

The parameters are
- $f_0 \in (0, +\infty)$: resonance frequency,
- $Q_L \in (0, +\infty)$: loaded quality factor,
- $\theta_{21} \in (-\pi, \pi]$: phase of $S_{21}$ at resonance,
- $\theta_{11} \in (-\pi, \pi]$: phase of $S_{11}$ at resonance,
- $A \in (0, +\infty)$: amplification,
- $k_1 \in (0, +\infty)$: coupling coefficient for port 1,
- $k_2 \in (0, +\infty)$: coupling coefficient for port 2.

It's useful to take the modulus and the phase of equations $(1)$ and $(2)$.
$$
|S_{11}|(f) = A\, \sqrt{\frac{B^2 + Q_L^2\delta^2}{1 + Q_L^2\delta^2}} \tag{5.A}
$$
$$
\operatorname{Arg} S_{11}(f) = \operatorname{Arg}\left\{ e^{j\theta_{11}} \frac{B + jQ_L\delta}{1 + jQ_L\delta} \right\} \tag{5.B}
$$
$$
|S_{21}|(f) = \frac{AC}{\sqrt{ 1 + Q_L^2\delta^2 }} \tag{6.A}
$$
$$
\operatorname{Arg} S_{21}(f) = \operatorname{Arg}\left\{ \frac{e^{j\theta_{21}}}{1 + jQ_L\delta} \right\} \tag{6.B}
$$
For equation $(6.B)$, we used the fact that $C > 0$ for every choice of $k_1$ and $k_2$ in their domains. Notice how the sign of $B$ is not predetermined.

## Fitting consideration
- explain why we added $A$, $\theta_{11}$ and $\theta_{21}$.
- necessity of having $A_{11} = \alpha A_{21}$, with known $\alpha$, for independence (for us $\alpha = 1$).
- why we fit for $Q_L$ and not $Q_i$.
- why we are forced to do a simultaneous fit
- observe that $C > 0$ but $B$ can be negative
- we focus on one resonance of the cavity

## Parameter estimation
We want to estimate the above mentioned parameters starting from the VNA measurements of $S_{11}$ and $S_{21}$. For simplicity we must assume that both $S_{11}$ and $S_{21}$ are measured of the same set of frequencies $\{f^k\}_{k=1,\dots,N}$. So, our data is organized in two sets of complex datapoints:
$$
\{(f^k, S_{11}^k)\}_{k=1,\dots,N} \subset \mathbb{R}\times \mathbb{C},
$$
$$
\{(f^k, S_{21}^k)\}_{k=1,\dots,N} \subset \mathbb{R}\times \mathbb{C}.
$$

### Estimation of $f_0$
The maximum of $|S_{21}|$ is found at the frequency $f_\text{max}$ such that $\delta(f_\text{max}, f_0) = 0$, as directly implied by equation $(6.A)$ and the observation $A,C,Q_L^2,\delta^2>0$. This condition gives immediately $f_\text{max} = f_0$. We therefore estimate $f_0$ with
$$
\hat f_0 = f^{k_\text{max}},\, k_\text{max}\text{ such that }|S^{k_\text{max}}_{21}| = \max_i \{|S^i_{21}|\}.
$$
We assume that $k_\text{max}$ exists and is unique.

### Estimation of $Q_L$
We start by defining $\Delta f = f_0/Q_L$. Then we calculate
$$
\begin{aligned}
\delta(f_0 + k\Delta f) &= \frac{k}{Q_L} \frac{2 + k/Q_L}{1 + k/Q_L} \sim \frac{2k}{Q_L} \text{ for }{Q_L \to +\infty},
\end{aligned}
$$
which, together with equation (6.A), gives
$$
|S_{21}(f_0 + k\Delta f)| \sim \frac{|S_{21}(f_0)|}{\sqrt{1 + 4k^2}} \text{ for }{Q_L \to +\infty}.
$$
Assuming $Q_L \gg 1$, we can take the asymptotic behaviour as a good approximation. 
Looking for a value $f^k$ bigger than $f_0$ such that $|S_{21}^k| = |S_{21}(\hat{f}_0)|/ \sqrt{1 + 4k^2}$ we can estimate $\Delta f$ with
$$
\Delta \hat{f} = (f^k - \hat{f}_0)/k.
$$
Exploiting the definition of $\Delta f$ we get an estimate for $Q_L$:
$$
\hat{Q}_L = \frac{\hat{f}_0}{\Delta\hat{f}}.
$$

### Estimation of $\theta_{21}$
At resonance, $\delta = 0$, therefore $\operatorname{Arg}S_{21}(f_0) = \theta_{21}$.
Hence, we can estimate the parameter as
$$
\hat{\theta}_{21} = \operatorname{Arg}S_{21}^{k_\text{max}}.
$$

### Estimation of $B(k_1, k_2)$
We consider some $k \neq k_\text{max}$ and constrain the model $(5.B)$ to pass through the point:
$$
\operatorname{Arg} S_{11}^k
= 
\operatorname{Arg} S_{11} (f^k). 
$$ 
Dividing by $e^{j\theta_{11}}$ and taking the tangent of both sides we get
$$
\begin{aligned}
\tan \left( 
    \operatorname{Arg} \big\{S_{11}^k/e^{j\theta_{11}} \big\} 
\right)
&= \tan \left( 
    \operatorname{Arg} \big\{S_{11}(f^k)/e^{j\theta_{11}}\big\} 
\right) \\
&= \frac{
    \mathbb{I}\text{m} \big\{S_{11}(f^k)/e^{j\theta_{11}}\big\}
}{
    \mathbb{R}\text{e} \big\{S_{11}(f^k)/e^{j\theta_{11}}\big\}
} \\
&= \frac{
    \delta(f^k) Q_L\cdot( 1 - B)
}{
    (\delta(f^k) Q_L)^2 + B
}.
\end{aligned}
$$
We isolate $B$.
$$
B(k_1, k_2) = \frac{
    1 - \tan \left( 
    \operatorname{Arg} \big\{ S_{11}^k/ e^{j\theta_{11}} \big\} 
\right) \cdot [\delta(f^k, f_0) Q_L]
}{
    1 + \tan \left( 
    \operatorname{Arg} \big\{ S_{11}^k/ e^{j\theta_{11}} \big\} 
\right) / [\delta(f^k, f_0) Q_L]
}
$$
To turn this identity into an estimate for $B$ we are missing an estimate for $(\theta_{11} \text{ mod } \pi)$ since
$$
\tan\left( \operatorname{Arg} \big\{ S_{11}^k/ e^{j\theta_{11}} \big\}\right) = 
\tan\left( \operatorname{Arg} \big\{ S_{11}^k/ e^{j(\theta_{11}\pm n\pi)} \big\}\right),\, \forall n \in \mathbb{N}.
$$
The key observation here is that an estimate of $(\theta_{11} \text{ mod } \pi)$ can be given without knowing $\operatorname{sign} B$, whereas an estimate of $\theta_{11}$ requires such knowledge. In fact, since
$$
\operatorname{Arg} S_{11}(f_0) = \operatorname{Arg}\big\{{Be^{j\theta_{11}}}\big\},
$$
a sign difference in $B$ gives a $\pm\pi$ displacement in the argument. Directly from the previous equation, without knowing $\operatorname{sign}B$, we estimate
$$
(\hat\theta_{11} \text{ mod } \pi) = \operatorname{Arg} S_{11}^{k_\text{max}}.
$$
With this we can _easily_ estimate $B(k_1, k_2)$:
$$
\hat B = \frac{
    1 - \tan \left( 
    \operatorname{Arg} \big\{ S_{11}^k/ e^{j\operatorname{Arg} S_{11}^{k_\text{max}}} \big\} 
\right) \cdot [\delta(f^k, \hat f_0) \hat Q_L]
}{
    1 + \tan \left( 
    \operatorname{Arg} \big\{ S_{11}^k/ e^{j\operatorname{Arg} S_{11}^{k_\text{max}}} \big\} 
\right) / [\delta(f^k, \hat f_0) \hat Q_L]
}
$$

### Estimation of $\theta_{11}$
We start by observing that
$$
\operatorname{Arg} S_{11}(f_0) = \operatorname{Arg}\big\{{\operatorname{sign}(B) \, e^{j\theta_{11}}}\big\} = \begin{cases} \theta_{11},&B > 0 \\ \theta_{11} \pm \pi,&B < 0\end{cases}
$$
Isolating $\theta_{11}$ yields
$$
\theta_{11} = \begin{cases} \operatorname{Arg} S_{11}(f_0),&B > 0 \\ \operatorname{Arg} S_{11}(f_0) \mp \pi,&B < 0.\end{cases}
$$
Hence our estimate for $\theta_{11}$ is 
$$
\hat\theta_{11} = \begin{cases} \operatorname{Arg} S_{11}^{k_{\text{max}}},&\hat B > 0 \\ \operatorname{Arg} S_{11}^{k_{\text{max}}} \mp \pi,&\hat B < 0.\end{cases}
$$
The freedom of choice between $-\pi$ and $+\pi$ is fixed by the convention ${\theta_{11} \in (-\pi, \pi]}$. 

### Estimation of $A$
We start by observing that $|S_{11}|(f_0) = A|B|$.
An estimate for $A$ is straightforward:
$$
\hat A = {|S_{11}^{k_\text{max}}|}/{|\hat B|}.
$$

### Estimation of $C(k_1, k_2)$
We start by observing that $|S_{21}|(f_0) = AC$.
An estimate for $C$ is straightforward:
$$
\hat C = {|S_{21}^{k_\text{max}}|}/{\hat A}.
$$

### Estimation of $k_1$ and $k_2$
Since we have an estimate for $B(k_1, k_2)$ and $C(k_1, k_2)$, we can invert the relations $(3)$ and $(4)$ to isolate $k_1$ and $k_2$. Defining $\alpha = (1-B) / (1+B)$
$$
\begin{aligned}
k_1 &= 4\left[4\alpha - C^2(\alpha+1)^2\right]^{-1}, \\
k_2 &= \alpha k_1 -1.
\end{aligned}
$$
Estimates for $k_1$ and $k_2$ come naturally
$$
\begin{aligned}
\hat k_1 &= 4\left[4\hat \alpha - \hat C^2(\hat \alpha+1)^2\right]^{-1}, \\
\hat k_2 &= \hat \alpha \hat k_1 -1.
\end{aligned}
$$
where $\hat \alpha = (1-\hat B) / (1+\hat B)$.


