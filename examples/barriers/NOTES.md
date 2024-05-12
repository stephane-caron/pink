
## Control Barrier Functions

Control barrier functions (CBFs) are used to enforce safety for nonlinear control affine systems of the form:

$$
\dot{x}=f(x)+g(x) u
$$

where $f$ and $g$ are locally Lipschitz, $x \in D \subset \mathbb{R}^{n}$, and $u \in U \subset \mathbb{R}^{m}$ is the set of admissible inputs.

**Safety and Safe Sets**

Safety is framed in the context of enforcing invariance of a set, i.e., not leaving a safe set. Consider a set $\mathcal{S}$ defined as the superlevel set of a continuously differentiable function $h: D \subset \mathbb{R}^{n} \rightarrow \mathbb{R}$:

$$
\begin{aligned}
\mathcal{S} & =\left\{x \in D \subset \mathbb{R}^{n}: h(x) \geq 0\right\} \\
\partial \mathcal{S} & =\left\{x \in D \subset \mathbb{R}^{n}: h(x)=0\right\} \\
\operatorname{Int}(\mathcal{S}) & =\left\{x \in D \subset \mathbb{R}^{n}: h(x)>0\right\}
\end{aligned}
$$

We refer to $\mathcal{S}$ as the safe set. The system is safe with respect to the set $\mathcal{S}$ if the set $\mathcal{S}$ is forward invariant. Formally, the set $\mathcal{S}$ is forward invariant if for every $x_{0} \in \mathcal{S}$, the solution $x(t)$ satisfying $x(0)=x_0$ remains in $\mathcal{S}$ for all $t$.

**Control Barrier Functions**

A function $h$ is a control barrier function (CBF) if there exists an extended class $\mathcal{K}_{\infty}$ function $\alpha$ such that:

$$
\sup _{u \in U}\left[L_{f} h(x)+L_{g} h(x) u\right] \geq-\alpha(h(x))
$$

for all $x \in D$. Here $L_f h$ and $L_g h$ denote the Lie derivatives of $h$ along $f$ and $g$.

The set of all control values that render $\mathcal{S}$ safe is:

$$
K_{\mathrm{cbf}}(x)=\left\{u \in U: L_{f} h(x)+L_{g} h(x) u+\alpha(h(x)) \geq 0\right\}
$$

The main result regarding CBFs is that their existence implies the control system is safe:

**Theorem:** Let $\mathcal{S} \subset \mathbb{R}^{n}$ be a set defined as the superlevel set of a continuously differentiable function $h: D \subset \mathbb{R}^{n} \rightarrow \mathbb{R}$. If $h$ is a CBF on $D$ and $\frac{\partial h}{\partial x}(x) \neq 0$ for all $x \in \partial \mathcal{S}$, then any Lipschitz continuous controller $u(x) \in K_{\mathrm{cbf}}(x)$ renders the set $\mathcal{S}$ safe and asymptotically stable in $D$.

**Optimization Based Control**

To synthesize safety-critical controllers, we can solve a quadratic program (QP) to minimally modify an existing controller $k(x)$ to guarantee safety:

$$
\begin{aligned}
u(x)=\underset{u \in \mathbb{R}^{m}}{\operatorname{argmin}} & \frac{1}{2}\|u-k(x)\|^{2} \quad(\mathrm{CBF}-\mathrm{QP}) \\
\text { s.t. } & L_{f} h(x)+L_{g} h(x) u \geq-\alpha(h(x))
\end{aligned}
$$

When there are no input constraints ($U=\mathbb{R}^m$), the CBF-QP has a closed-form solution given by the min-norm controller.

**Discrete-Time Implementations**

Discrete-time implementations of continuous-time CBF-based safety filters can lead to some challenges. When solving the CBF-QP at a sampling time $\Delta t>0$, safety is only guaranteed at the initial time step $t_{0}$, but not for the open time interval $(t_{0}, t_{0}+\Delta t)$. Suboptimal performance can arise when $\|L_{g} L_{f}^{s-1} h(x)\| \rightarrow 0$, as large inputs permissible by the CBF condition applied over the finite time interval could result in undesirable behavior.

Another issue occurs when $L_{g} L_{f}^{s-1} h(x)=0$, indicating a local relative degree higher than $s$. In such cases, the safety controller becomes inactive, allowing potentially unsafe control inputs $k(x)$ to be applied for at least the time interval $[t_{0}, t_{0}+\Delta t)$, which may lead to safe set violations or suboptimal performance.

One method to handle the case of $\|L_{g} L_{f}^{s-1} h(x)\| \rightarrow 0$ is by modifying the safety filter objective function. We can add a term that explicitly accounts for $L_{g} L_{f}^{s-1} h(x)$ becoming close to 0. The proposed modified safety filtering objective is:

$$
J(x)=\frac{1}{2}\|u-k(x)\|^{2}+\frac{r}{2\|L_{g} L_{f}^{s-1} h(x)\|^{2}}\|u-k_{\text{safe}}(x)\|^{2}
$$

where $k_{\text{safe}}$ is a known safe backup control policy (e.g., a stabilizing controller that renders $\mathcal{S}$ control invariant). The parameter $r>0$ is a weighting factor.

This new objective replaces the standard CBF-QP objective for all $\|L_{g} L_{f}^{s-1} h(x)\|>\epsilon$, where $\epsilon$ is a small positive number. The closer $\|L_{g} L_{f}^{s-1} h(x)\|$ gets to 0, the greater the impact of the second term in the safety filtering objective. In this case, the safety filter will track the safe backup control policy instead of the potentially unsafe control policy $k(x)$. The weighting parameter $r$ determines the balance between the two terms when $\|L_{g} L_{f}^{s-1} h(x)\|$ is far from 0. To avoid numerical instabilities, we set $u(x)=k_{\text{safe}}(x)$ when the system is in a state $x$ such that $\|L_{g} L_{f}^{s-1} h(x)\| \leq \epsilon$.

The modified CBF-QP with the penalty term becomes:

$$
\begin{aligned}
u(x)=\underset{u \in \mathbb{R}^{m}}{\operatorname{argmin}} & \frac{1}{2}\|u-k(x)\|^{2}+\frac{r}{2\|L_{g} L_{f}^{s-1} h(x)\|^{2}}\|u-k_{\text{safe}}(x)\|^{2} \\
\text{s.t.} & L_{f} h(x)+L_{g} h(x) u \geq-\alpha(h(x))
\end{aligned}
$$

This strategy requires almost no additional computational effort. However, in practice, the design of the safe backup control policy $k_{\text{safe}}$ will require some attention. The backup policy should be able to return the system to states where $\|L_{g} L_{f}^{s-1} h(x)\|>\epsilon$. Otherwise, the system will continue using the backup control policy $k_{\text{safe}}$ for all future time.


## CBF in Differential IK

Let us study how CBF theory can be applied to solve the multitask inverse kinematics problem subject to configuration-based nonlinear inequalities. 

Mathematically, our goal is to find the configuration motion $\boldsymbol{q}(t) \in \mathcal{C}$ such that:

$$
\boldsymbol{r}_i(\boldsymbol{q})=\boldsymbol{p}_i^{*}-\boldsymbol{p}_i(\boldsymbol{q})
$$

where $\boldsymbol{p}_i^{*}$ is the desired task reference, $\boldsymbol{r}_i(\boldsymbol{q})$ is the task residual error, and $\boldsymbol{p}_i(\boldsymbol{q})$ is the forward kinematics function mapping configuration variables to task space.

In addition to the kinematic tasks, suppose we want to enforce the following nonlinear safety constraints:

$$
h_j(\boldsymbol{q}) \geq 0
$$

Directly solving this constrained optimization problem can lead to challenging nonlinear programs. However, in the context of differential inverse kinematics, we can leverage CBFs to reformulate the problem as a quadratic program (QP) with linear inequality constraints.

**CBF QP Formulation**

Differentiating the kinematic task errors once and introducing stabilizing feedback gains yields the following QP formulation:

$$
\underset{\dot{\boldsymbol{q}}}{\operatorname{minimize}} \sum_{\operatorname{task} i} w_{i}\left\|\boldsymbol{J}_{i} \dot{\boldsymbol{q}}-K_{i} \boldsymbol{v}_{i}\right\|^{2}
$$

Here, $\boldsymbol{J}_i$ is the task Jacobian, $K_i$ is a positive definite feedback gain matrix, and $\boldsymbol{v}_i$ is an auxiliary control input. By treating joint velocities $\dot{\boldsymbol{q}}$ as optimization variables, we can incorporate the safety constraints $h_j(\boldsymbol{q}) \geq 0$ as CBF conditions:

$$
\dot{h}_j(\boldsymbol{q})+\alpha_j(h_j(\boldsymbol{q})) = \frac{\partial h_j}{\partial \boldsymbol{q}} \dot{\boldsymbol{q}} +\alpha_j(h_j(\boldsymbol{q})) \geq 0
$$

Enforcing this constraint ensures the system remains within the safe set $\mathcal{S}_j =\left\{\boldsymbol{q} \in D \subset \mathbb{R}^{n}: h_j(\boldsymbol{q}) \geq 0\right\}$. The extended class $\mathcal{K}$ function $\alpha_j$ provides a safety margin.

Combining the kinematic tasks and CBF constraints, we arrive at the differentiable IK optimization problem with safety guarantees:

$$
\begin{aligned}
& \underset{\dot{\boldsymbol{q}}}{\operatorname{minimize}} \sum_{\text {task } i} w_{i}\left\|\boldsymbol{J}_{i}(\boldsymbol{q}) \dot{\boldsymbol{q}}-K_{i} \boldsymbol{v}_{i}\right\|^{2} + \gamma(\boldsymbol{q})\left\| \dot{\boldsymbol{q}}-\dot{\boldsymbol{q}}_{safe}(\boldsymbol{q})\right\|^{2} \\
& \text { subject to: } \frac{\partial \boldsymbol{h}_j}{\partial \boldsymbol{q}} \dot{\boldsymbol{q}} +\alpha_j(\boldsymbol{h}_j(\boldsymbol{q})) \geq 0, \quad \forall j
\end{aligned}
$$

The configuration-dependent weight $\gamma(\boldsymbol{q})$, often chosen as $(\|\partial h_j \|  +\epsilon)^{-1}w_h$, ensures safety takes precedence when close to constraint boundaries. The safe backup policy $\dot{\boldsymbol{q}}_{safe}(\boldsymbol{q})$ provides a fallback when $\|\partial h_j \|$ approaches zero. A simple choice is the zero-velocity policy $\dot{\boldsymbol{q}}_{safe} = \boldsymbol{0}$, which stops the robot. Alternatively, one can stabilize to a safe initial configuration.

This general CBF-based framework enables the encoding of various safety behaviors through appropriate choices of $h_j(\boldsymbol{q})$, such as:

1. Joint Limits:
$$h_j(\boldsymbol{q}) = \begin{cases} 
      q_j - q_j^{min} & j = 1,\dots,n \\
      q_j^{max} - q_j & j = n+1,\dots,2n
   \end{cases}$$

2. End-Effector Position Barriers:
$$h_j(\boldsymbol{q}) = \|\boldsymbol{p}_{EE}(\boldsymbol{q}) - \boldsymbol{p}_{obs}\| - d_{safe}$$

3. Self-Collision Avoidance (between links $i$ and $k$):  
$$h_j(\boldsymbol{q}) = \operatorname{sd}(\mathcal{B}_i(\boldsymbol{q}), \mathcal{B}_k(\boldsymbol{q})) - d_{safe}$$

Here, $\operatorname{sd}(\cdot)$ denotes the signed distance between two geometric shapes $\mathcal{B}_i$ and $\mathcal{B}_k$ representing robot links.  
