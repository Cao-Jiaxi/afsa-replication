# Reinforcement Learning lecture (AMS 520, Fall 2026): the deck condensed

Prof. Polak's deck *Reinforcement Learning: Foundations, control, and modern post training* (`RL_Fall2026.pdf`, 577 slides, dated 2026-11-10; course material, not in the repo), condensed to study notes. Every item cites its slide as pN. How these ideas apply to AFSA is in [EXPLORATION.md](EXPLORATION.md).

**How the notes were made.**
- The text was extracted with `pdftotext`. The deck has no overlay pages, so every slide is distinct.
- Sections 8–11 and 17–19, the ones closest to the project, were condensed by hand. The others were condensed by four agents and spot-checked against the slide text.
- Formulas were rebuilt from flattened text where needed. A few are marked "(reconstructed)".
- Figure-only slides are covered by their captions: p30, p177, p183, p187, p195, p215, p255, p346 and p353.

## The deck in eleven lines

The deck's recurring point: **a guarantee holds only in its stated setting** (data oracle, normalization, coverage, policy class), so state the claim before the rate.

1. **The problem (§1).** In sequential decisions, actions change later states. The reward is a modelling choice, and Bellman equations need a Markov state; without one the problem is a POMDP.
2. **Exploration (§2).** Regret measures the cost of learning while acting. Fixed ε pays linear regret. UCB and Thompson sampling get logarithmic regret for a fixed gap and $\sqrt{KT\log T}$ without one, against the Lai–Robbins and minimax lower bounds.
3. **Exact dynamic programming (§3–4).** Bellman operators are γ-contractions, which gives policy and value iteration, the occupancy-measure LP and the performance difference lemma. Errors are amplified by powers of $1/(1-\gamma)$, and every complexity rate belongs to its data oracle: generative, online or offline.
4. **Model-free prediction and control (§5–6).** Monte Carlo targets are unbiased but noisy; TD targets bootstrap and inherit the critic's bias. SARSA and Q-learning converge under visit-indexed step sizes. The max causes an optimistic bias, which double Q-learning addresses. Importance sampling needs support, and its variance can grow like $p^{-H}$.
5. **Function approximation (§7).** Approximation, bootstrapping and off-policy data together can diverge (the deadly triad). Realizability, Bellman completeness, coverage and exploration are separate assumptions. DQN's tricks work without general proofs.
6. **Policy optimization (§8–9).** The deck covers score-function gradients and baselines, actor-critic methods, and the deterministic policy gradient. The natural gradient is the KL trust region in Fisher geometry. TRPO, PPO (clipping bounds no KL), GAE and SAC follow.
7. **Models and logged data (§10–11).** Dyna, MCTS and Dreamer plan with a model, but model error grows like $H^2$ with the horizon. Offline RL is limited by coverage: an unlogged action's value is unidentified, so pessimism is the defence (CQL, IQL, MOPO), followed by hybrid fine-tuning.
8. **Demonstrations and preferences (§12–15).**
   - Learning from demonstrations: behaviour cloning compounds as $H^2\epsilon$, which DAgger fixes; maximum-entropy inverse RL; diffusion policies.
   - Learning from preferences: Bradley–Terry; the KL-regularized optimum $\pi_{ref}e^{r/\beta}$; DPO and its relatives.
   - RL with verifiers (the GRPO family) and multi-turn credit assignment.
9. **Structured problems (§16–17).**
   - Markov games, with centralized training and decentralized execution.
   - LQR: Riccati recursion, linear feedback, and noise that changes only the constant; policy gradients for LQR.
10. **Finance (§18).**
    - The portfolio MDP's state includes the holdings, and the reward must match the mandate.
    - Static CVaR is not Bellman-recursive.
    - Every input needs a date (a timing contract), and counterfactual fills need assumptions.
11. **Evaluation (§19–20).**
    - Separate training randomness from evaluation noise, and match budgets.
    - Off-policy evaluation: importance sampling, the direct model and the doubly robust estimator.
    - Split train, selection and test chronologically, with the selection frozen before the test.
    - A discovered algorithm needs held-out problems.

## Front matter (p1–p5)
**Learning goals (p2).** Formulate sequential decisions with explicit information and reward conventions; derive value updates and policy improvement rules; explain how approximation, exploration and data coverage affect guarantees; compare planning, imitation and LLM post training under stated assumptions; design dated financial experiments with measurable costs and uncertainty; evaluate research claims against their data access and compute budgets.
**Course structure (p3–p5), 21 sections.** 1 RL problem; 2 Bandits and exploration; 3 Finite MDPs; 4 DP guarantees; 5 Monte Carlo and TD prediction; 6 Control and off policy learning; 7 Function approximation and deep value learning; 8 Policy gradients and actor critic; 9 Trust regions and entropy regularized control; 10 Planning and learned world models; 11 Offline and hybrid RL; 12 Imitation, inverse RL, generative policies; 13 Preference learning and RLHF; 14 Reasoning and verifiable rewards; 15 Multi turn agents and value based post training; 16 Multi agent RL; 17 Linear quadratic control; 18 Sequential decisions in quantitative finance; 19 Evaluation, reproducibility, deployment; 20 Algorithm discovery and frontiers; 21 Mathematical appendix and extended proofs.

## 1. The reinforcement learning problem (p6–p34)
**Core idea.** RL studies agents that learn to maximize expected cumulative reward by interacting with an environment; unlike prediction, the policy is the decision, actions change later states, and the data distribution depends on the policy (p7–p8).
**Definitions and results.**
- Prediction vs decision (p7): a forecast is an input to a decision; credit assignment asks which earlier decisions caused later outcomes. Finance link: order execution, where trading now reduces inventory and changes the cost of finishing later.
- Agent-environment loop (p9): observe state, choose action by policy, observe reward and next state, repeat.
- Reward hypothesis (p10): many objectives can be written as expected cumulative reward, but choosing that representation is part of modeling; risk limits, hard constraints and multiple objectives may need extra structure; a high proxy reward can coexist with poor real outcomes. Finance link: expected profit alone does not encode inventory limits or drawdown tolerance.
- Return (p11–p13): $G_t=\sum_{k\ge0}\gamma^k R_{t+k+1}=R_{t+1}+\gamma G_{t+1}$, $\gamma\in[0,1)$. $\gamma$ read as continuation probability, time preference, or a device ensuring $E[G_t]<\infty$ for bounded rewards; finite episodes may use $\gamma=1$.
- Terminal and time limits (p14): target $Y_t=R_{t+1}+\gamma m_t\hat V(S_{t+1})$ with $m_t=0$ only after a true terminal transition; a data-collection time limit need not terminate the task; if a deadline defines the task, put remaining time in the state; $V_{H+1}=0$.
- Value functions (p15–p17): $V^\pi(s)=E_\pi[G_t|S_t=s]$, $Q^\pi(s,a)=E_\pi[G_t|S_t=s,A_t=a]$, $V^\pi=\sum_a\pi(a|s)Q^\pi$; $V^*=\max_\pi V^\pi$, $Q^*=\max_\pi Q^\pi$, $V^*=\max_a Q^*$.
- Bellman expectation (p18–p20): $V^\pi(s)=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V^\pi(s')]$, with the matching $Q^\pi$ form (p19); matrix form $V^\pi=r^\pi+\gamma P^\pi V^\pi$, so $V^\pi=(I-\gamma P^\pi)^{-1}r^\pi$.
- Bellman optimality (p21–p22): $V^*(s)=\max_a\sum p[r+\gamma V^*(s')]$, $Q^*(s,a)=\sum p[r+\gamma\max_{a'}Q^*(s',a')]$; the max makes it nonlinear; $Q^*$ can be learned from experience without a model.
- Agent components (p23): policy, value function, model; not every agent has all three.
- Markov property (p25): $\Pr(S_{t+1},R_{t+1}|S_t,A_t,H_t)=\Pr(S_{t+1},R_{t+1}|S_t,A_t)$; this is what makes the Bellman equations valid. If it fails, the problem is a POMDP ($O_t\ne S_t$; recurrent networks, belief tracking) (p24, p26). The course focuses on MDPs.
- Agent state options (p24): history, raw observation, learned features, Markov state (sufficient statistic of history).
- Maze example (p27–p29): reward +1 at goal, 0 elsewhere, $\gamma=0.9$; optimal values rise near the goal. p30 is a generic MDP illustration (figure only).
- Taxonomy (p31–p33): value-based (Q-learning, SARSA) vs policy-based (REINFORCE, PPO) vs actor-critic hybrids; model-free (simpler, less data efficient) vs model-based (MCTS, Dyna, MuZero; more sample efficient in theory, but model learning is hard); prediction (evaluate $V^\pi$) vs control (find $\pi^*$), prediction often a subroutine of control.
**Algorithms.** None; p34 summary lists formulation, value functions, Bellman equations, components, Markov property.
**Caveats the deck stresses.**
- Reward specification is a modeling choice; proxy reward can mislead (p10).
- Do not terminate bootstrapping at time limits; include remaining time in the state if a deadline defines the task (p14).
- Bellman validity rests on the Markov state assumption (p25).

## 2. Bandits and exploration (p35–p116)
**Core idea.** Bandits isolate learning while acting with no training data: balance exploration against exploitation, measured by regret. Guarantees (fixed-gap vs minimax vs Bayesian) answer different questions and must be compared only under matching reward models and quantifiers (p46, p112, p116).
**Definitions and results.**
- Motivation (p36): clinical trials, A/B tests, ads, and portfolio allocation (learn returns while managing risk).
- Protocol (p37–p39): each round pick $A_t\in\{1..K\}$, observe $R_t\sim\nu_{A_t}$; $q_*(a)=E[R|A=a]$, $q^*=\max_a q_*(a)$, arm suboptimal if $q_*(a)<q^*$.
- Sample average (p40–p41): $Q_t(a)=\sum_\tau R_\tau 1(A_\tau=a)/N_{t-1}(a)$; converges a.s. by the LLN, error $O(1/\sqrt N)$.
- Regret (p43–p45): $\mathrm{Regret}(T)=Tq^*-E\sum_t R_t=\sum_a\Delta_a E[N_a(T)]$, $\Delta_a=q^*-q_*(a)$ deterministic.
- Incremental update (p47–p49): $Q_{n+1}=Q_n+\frac1{n+1}(R_{n+1}-Q_n)$ is unbiased; fixed $\alpha$ gives exponential recency weights $\alpha(1-\alpha)^{n-i}$ (large $\alpha$ tracks fast but noisy). Use $1/n$ for stationary, fixed $\alpha$ for non-stationary.
- Regret bound types (p46): fixed instance (gaps fixed as $T$ grows); worst case (sup over instances, may depend on $T$); Bayesian (prior average). An algorithm upper bound and a model-class lower bound are distinct statements.
- Fixed $\epsilon$ cost (p54–p55): $E[N_a^{explore}(T)]=\epsilon T/K$; example means 0.6 and 0.5, $\epsilon=0.1$: exploration regret $0.005T$, i.e. 50 at $T=10{,}000$, linear in $T$. Decay requirement $\epsilon_t\to0,\ \sum\epsilon_t=\infty$ gives consistency but no finite-time rate (p56).
- Lai-Robbins (p62–p63, p66): for Bernoulli arms, unique best mean, uniformly efficient policy (subpolynomial regret across the class): $\liminf E R_T/\log T\ge\sum_{a:\Delta_a>0}\Delta_a/\mathrm{kl}(p_a,p^\star)$. Instance-dependent, asymptotic. Small KL means hard to distinguish, so more sampling.
- KL (p64–p65): $KL(P\|Q)=E_P\log(P/Q)\ge0$, asymmetric; Bernoulli 0.4 vs 0.5 gives about 0.0223.
- Minimax (p67–p68): for bounded $K$-armed bandits, $T\ge K$: $\inf_A\sup_\nu E R_T\ge c\sqrt{KT}$; hard gaps shrink with $T$. Compatible with logarithmic fixed-gap regret; do not label an arbitrary gap-free bound as the minimax rate.
- Hoeffding confidence radius (p69–p71): rewards in [0,1], fixed count $n$: $P(|\hat\mu_n-\mu|>c)\le2e^{-2nc^2}$, $c_n(\delta)=\sqrt{\log(2/\delta)/2n}$. Repeated or adaptive use needs a simultaneous argument; p75 gives a union budget $\delta_{a,n}=\delta/(Kn(n+1))$, radius $\sqrt{\log(2Kn(n+1)/\delta)/2n}$, all intervals hold with prob $\ge1-\delta$ (a confidence sequence, distinct from UCB1's expected-regret accounting).
- UCB1 gap-dependent regret (p80): bounded independent stationary rewards, stated initialization: $ER_T\le 8\sum_{a:\Delta_a>0}\log T/\Delta_a+C\sum_a\Delta_a$. Logarithmic, but lacks the KL constant of Lai-Robbins (p81).
- UCB1 gap-free (p82): $ER_T=O(\sqrt{KT\log T}+K)$, from splitting arms by gap threshold. It is a $\sqrt{\log T}$ factor off the minimax benchmark; the factor depends on the algorithm variant.
- Thompson sampling rates (p92–p93): Bernoulli, Beta posterior, fixed instance: $ER_T\lesssim\sum\Delta_a\log T/\mathrm{kl}(p_a,p^\star)$; problem-independent $O(\sqrt{KT\log T})$ (Agrawal and Goyal 2013). State prior and regularity assumptions; not a guarantee for every posterior-sampling implementation. p94 claims Thompson matches Lai-Robbins and is often better empirically than UCB, but needs a prior.
- Linear bandits (p98–p100, p108): $R_t=\phi(A_t)^\top\theta_*+\eta_t$, sub-Gaussian noise; for huge or continuous action sets; shares evidence across actions via features (risk: linearity and feature adequacy).
- Optimistic linear regret (p106): bounded features and parameter norm, conditionally sub-Gaussian noise, valid confidence radii: $\tilde O(d\sqrt T)$, depends on dimension $d$ not arm count; picking the optimistic action needs an optimization oracle, hard for continuous sets.
- Explore-then-commit (p109–p111): $m$ pulls per arm ($Km\le T$), then commit; fixed-budget uniform ETC gives $\tilde O(K^{1/3}T^{2/3})$ gap-free, weaker than $\sqrt{KT}$; no recovery from a mistaken commitment.
- Comparison table (p112): fixed $\epsilon$ linear; ETC $\tilde O(K^{1/3}T^{2/3})$; UCB1 $O(\sqrt{KT\log T}+K)$; Beta-Bernoulli TS $O(\sqrt{KT\log T})$ in the analyzed setup; minimax lower $\Omega(\sqrt{KT})$ for $T\ge K$.
**Algorithms.**
- Greedy (p50–p52): $A_t=\arg\max Q_t(a)$; fails by locking onto a suboptimal arm after early luck (arms 0.10 vs 0.09 example). Must explore.
- $\epsilon$-greedy (p53–p57): random arm w.p. $\epsilon$, else greedy; uniform exploration ignores uncertainty; fixed $\epsilon$ pays linear regret; decaying schedules need their own analysis.
- Softmax/Boltzmann (p58–p60): $\pi_\tau(a)=e^{Q_t(a)/\tau}/\sum e^{Q_t(a')/\tau}$; $\tau\to0$ greedy, $\tau\to\infty$ uniform; K=3, Q=[1,0.5,0.2]: $\tau=0.1$ gives about [0.99,0.01,0], $\tau=1$ [0.66,0.24,0.10], $\tau=10$ [0.35,0.33,0.32]. Smooth and differentiable, so usable with gradient methods.
- UCB1 (p72–p79, p83): $A_t=\arg\max_a[Q_t(a)+\sqrt{2\ln t/N_t(a)}]$; untried arms have infinite bonus; no $\epsilon$ needed; $O(K)$ per step; bonus constant tunable. Tail $P(\hat\mu-\mu>c_t(n))\le t^{-4}$ (p74). Needs bounded, independent, stationary rewards.
- Thompson sampling (p84–p97): sample parameters from posterior, play the best sampled arm, update by Bayes rule; probability matching, $P(A_t=a)\approx P(a\text{ optimal}|\text{history})$ (p90). Beta-Bernoulli: prior Beta(1,1), sample $\tilde p_a$, play argmax, increment $\alpha$ on success or $\beta$ on failure (p86–p89); conjugacy makes the update tractable. Extensions: Gaussian, contextual (posterior over weights), general likelihoods via MCMC or variational inference (p96). Regret stays $O(\log T)$ despite posterior concentration (p95).
- LinUCB (p101–p107): ridge $\hat\theta_t=(X^\top X+\lambda I)^{-1}X^\top y$, $V_t=\lambda I+\sum\phi\phi^\top$ (Sherman-Morrison updates); play $\arg\max_a\phi(a)^\top\hat\theta_t+\alpha\sqrt{\phi(a)^\top V_t^{-1}\phi(a)}$, the optimism principle over the ellipsoid $\|\theta-\hat\theta\|_{V_t}\le$ radius (p103–p105, p107); ellipsoid semiaxes $\beta/\sqrt{\lambda_i}$ (p104).
**Caveats the deck stresses.**
- Compare bounds only under matching reward model, quantifiers and comparison criterion (p46, p112, p116).
- Fixed-gap logarithmic results and the minimax $\sqrt{KT}$ do not conflict (p67–p68).
- Hoeffding at a fixed count is not an adaptive guarantee (p70–p71).
- Thompson claims depend on prior and setup (p92–p93).
- Pick method by credible reward assumptions (stationarity, independence, drift), computability of the optimistic action, and whether the metric is cumulative regret, identification or a constrained objective (p113, p116).

## 3. Finite Markov decision processes (p117–p138)
**Core idea.** A finite MDP $(S,A,p,r,\gamma)$ with the Markov property and full observability (p118); the Bellman operators give evaluation, improvement, policy iteration and value iteration, and an occupancy-measure (LP) view.
**Definitions and results.**
- Returns (p119): discounted, episodic (finite sum), average ($\frac1H\sum$); focus is discounted. Policies: deterministic, stochastic, stationary; an optimal policy can be taken deterministic and stationary in the discounted setting (p120).
- Bellman expectation derivation (p121) and matrix form (p122): $V^\pi=(I-\gamma P^\pi)^{-1}r^\pi$, invertible since $\gamma P^\pi$ has spectral radius $<1$.
- Bellman optimality (p123): finite discounted, bounded rewards: $V^\star(s)=\max_a E[R_{t+1}+\gamma V^\star(S_{t+1})|S_t=s,A_t=a]$; the realized $G_{t+1}$ need not equal $V^\star(S_{t+1})$, only its conditional expectation; a maximizing action gives a deterministic stationary optimal policy.
- Policy improvement theorem (p124–p125): $\pi'(s)=\arg\max_a Q^\pi(s,a)$ gives $V^{\pi'}\ge V^\pi$ everywhere; proof via monotone $T^{\pi'}$: $T^{\pi'}V^\pi=T^\star V^\pi\ge V^\pi$, iterate, contraction gives $(T^{\pi'})^nV^\pi\to V^{\pi'}$ (bounded rewards, $\gamma<1$).
- Worked example (p126): $\gamma=0.9$, two states, stay policy $r=(1,2)$, $P=I$, $v=(10,20)$; switching in state 1 gives $Q=0+0.9\cdot20=18>10$; improved values (18,20).
- Termination of policy iteration (p132–p133): finite discounted MDP, exact evaluation, deterministic stationary policies, fixed tie rule (keep current action if already greedy): at most $|A|^{|S|}$ policies, strict improvement cannot revisit a policy, stable greedy policy satisfies Bellman optimality. Boundedness plus monotonicity alone does not prove finite termination (e.g. $1-2^{-n}$); finiteness of the policy set supplies the step.
- Occupancy measure (p134–p135): $d^\pi_\mu(s,a)=(1-\gamma)\sum_t\gamma^tP(S_t=s,A_t=a|\mu,\pi)$, sums to one; $J_\mu(\pi)=\frac1{1-\gamma}\sum_{s,a}d^\pi_\mu(s,a)r(s,a)$. Flow constraint (p136): $\rho(s')=(1-\gamma)1(s'=s_0)+\gamma\sum_{s,a}\rho(s,a)p(s'|s,a)$.
- Value LP and dual (p137): $\min_v\sum\mu(s)v(s)$ s.t. $v(s)\ge r+\gamma\sum P v(s')$ for all $(s,a)$; dual $\max_{x\ge0}\sum x\,r$ s.t. flow constraint; $x=d^\pi_\mu/(1-\gamma)$ is the unnormalized occupancy (matching scales needed).
- Performance difference lemma (p138): shared initial distribution: $J(\pi')-J(\pi)=\frac1{1-\gamma}E_{s\sim d^{\pi'},a\sim\pi'}[A^\pi(s,a)]$; advantage under the old policy, state distribution of the new one; a surrogate using $d^\pi$ in place of $d^{\pi'}$ must control the distribution error (link to later trust-region methods).
**Algorithms.**
- Policy iteration (p127): evaluate $V^\pi$ (solve linear system), improve greedily, stop when policy is stable; finite in finite spaces, final policy optimal; evaluation can be expensive.
- Value iteration (p128–p130): $V_{k+1}(s)=\max_a\sum p[r+\gamma V_k(s')]$, one backup per step, converges to $V^*$ by contraction: $\|V_k-V^*\|_\infty\le\gamma^k\|V_0-V^*\|_\infty$; extract greedy policy at the end.
- Asynchronous DP (p131): in-place, prioritized sweeping, real-time DP; converges if each state is updated infinitely often.
**Caveats the deck stresses.**
- Policy iteration termination needs exact evaluation, tabular finiteness and consistent tie handling (p132–p133).
- Occupancy normalization must match between primal and dual (p137).
- Performance difference mixes old advantage with new state distribution (p138).

## 4. Dynamic programming guarantees (p139–p165)
**Core idea.** Bellman operators are $\gamma$-contractions in the max norm, giving unique fixed points and geometric convergence; approximation and sampling errors are amplified by horizon factors $(1-\gamma)^{-1}$ and beyond. Complexity results hold only for their stated oracle.
**Definitions and results.**
- Contraction and Banach (p140–p142): $\|Tx-Ty\|\le\gamma\|x-y\|$ on a complete space gives a unique fixed point, convergence from any $V_0$, rate $\gamma^k\|V_0-V^*\|$.
- Bellman optimality operator (p143–p144): $\|TV-TU\|_\infty\le\gamma\|V-U\|_\infty$, using $|\max_a u_a-\max_a v_a|\le\max_a|u_a-v_a|$ and that $P$ averages with mass one.
- Expectation operator $T^\pi$ (p145–p146): linear, same $\gamma$-contraction.
- Q operators (p147): both $T$ and $T^\pi$ on $Q$ contract; Q-learning iterates the optimality operator (off-policy).
- Approximate DP (p148–p150): $|V_{k+1}-TV_k|\le\epsilon_{k+1}$ gives $\|V_k-V^*\|_\infty\le\gamma^k\|V_0-V^*\|_\infty+\frac1{1-\gamma}\sum_j\gamma^{k-j-1}\epsilon_j$; constant error gives $O(\epsilon/(1-\gamma))$. Function approximation (p149) generalizes across states but risks non-contraction, instability and approximation error.
- Greedy policy bound (p151): $\|\tilde V-V^*\|_\infty\le\epsilon\Rightarrow\|V^{\tilde\pi}-V^*\|_\infty\le\frac{2\gamma}{1-\gamma}\epsilon$; large for $\gamma\approx1$.
- Sample complexity models (p152–p153): generative model (query any $(s,a)$) vs PAC (w.p. $1-\delta$, return $\ge V^*-\epsilon$); worst case over MDPs with given $|S|,|A|,\gamma$; idealized, ignores online exploration constraints.
- Simulation lemma (p154): $\|\hat p-p\|_1\le\delta$, $\|\hat r-r\|_\infty\le\delta$ gives $\|V^{\hat\pi}-V^*\|_\infty\le 2\gamma\delta/(1-\gamma)^2$ for the naive plug-in model-based approach.
- Generative-model rate (p155–p156): finite discounted MDP, rewards in [0,1]: $N=\tilde O(|S||A|/((1-\gamma)^3\epsilon^2))$ for a uniformly $\epsilon$-optimal policy (Sidford et al. 2018), matched by an $\Omega$ worst-case lower bound in the theorem's regime. $N$ counts transition samples, not episodes; the oracle can query states the policy never visits; not an online or offline guarantee; bounds must share reward normalization and accuracy definition.
- Return variance (p157): $|R_t|\le R_{\max}$ gives $|G_t|\le R_{\max}/(1-\gamma)$, $\mathrm{Var}(G_t)\le R_{\max}^2/(1-\gamma)^2$; independent rewards would give $\sigma^2/(1-\gamma^2)$. A range bound and an exact identity are different statements; correlated rewards need not follow the independent formula (p158).
- UCBVI regret (p160): $K$ episodes, horizon $H$, $T=KH$, variance-aware tabular (Azar, Osband, Munos 2017): leading term $\tilde O(\sqrt{H|S||A|T})=\tilde O(H\sqrt{|S||A|K})$, plus lower-order terms and confidence conditions; bonus constructions differ in horizon dependence. Proof sketch (p161): optimism $\hat V\ge V^*$ w.h.p., per-episode gap $V^*(s_0)-V^{\pi_k}(s_0)$, counting lemma; the slide states $\tilde O(\sqrt{H^3SAT})$.
- Bernstein bonus (p162–p163): variance-adaptive bonus $\tilde O(\sqrt{V(s,a)\log T/N}+\log T/N)$ is tighter for low variance, but unvisited actions, unknown rewards and lower-order terms remain; deterministic transitions alone do not fix a universal regret rate.
- Finite-horizon MDPs (p164): value $V^\pi(s,h)$ with $h$ steps remaining, no discounting; Bellman backup is not a contraction (factor 1), so different theory is needed.
- Data access table (p165): generative model (independent transitions at chosen state-action-time triples), online (trajectories from chosen exploration policies), offline (only the logged behavior distribution). Specify time-homogeneity and whether complexity counts episodes or steps; do not substitute $H=(1-\gamma)^{-1}$ into every theorem.
**Algorithms.**
- Approximate value iteration (p148): run the Bellman backup with errors; error bound as above.
- Finite-horizon optimistic VI (p159): $Q_{k,h}(s,a)=\min\{H-h+1,\ \hat r+\hat P V_{k,h+1}+b_{k,h}\}$, $V_{k,h}=\max_aQ_{k,h}$, $V_{k,H+1}=0$; the bonus must dominate the estimation error; the greedy policy collects the next episode.
**Caveats the deck stresses.**
- Greedy-policy error is amplified by $2\gamma/(1-\gamma)$; model error by $(1-\gamma)^{-2}$ (p151, p154).
- A complexity rate must be quoted with its oracle, normalization, accuracy regime and confidence statement (p158, p156).
- Function approximation breaks the contraction guarantee (p149).
- The generative-model rate is not an online or offline data guarantee (p155, p165).

## 5. Monte Carlo and temporal difference prediction (p166–p202)
**Core idea.** Without the model p(s',r|s,a), estimate V^π from sampled experience: Monte Carlo uses full returns, TD bootstraps one step ahead, n-step and λ-returns blend the two (p167–168).
**Definitions and results.**
- Prediction vs control (p167): prediction evaluates a given π; control finds π*.
- MC target (p169, p173–175): $V^\pi(s)=E_\pi[G_t|S_t=s]$. First-visit MC is unbiased and converges a.s. as visits → ∞ if returns are bounded, i.i.d. across episodes with finite variance (LLN). Returns repeated within one episode need not be independent (p174). High variance, since it waits for the episode to end (p173).
- TD target (p179–181): $\hat G_t=R_{t+1}+\gamma V(S_{t+1})$, error $\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$. Terminal transitions have mask 0 (p180). Convergence is a stochastic-approximation result under stated conditions.
- MC vs TD (p184–186): complete return is unbiased; TD target has expectation $T^\pi V$ with conditional bias $\gamma E[e(S')|s]$, $e=V-V^\pi$. Neither has universally smaller variance. Var(Y|s) includes a reward–successor covariance term (p185).
- Batch case (p189): batch MC fits observed returns by least squares; tabular batch TD solves the empirical Bellman consistency equations (an implied Markov model). They need not agree. Unbiased MC targets do not give an unbiased nonlinear fit.
- n-step return (p190–192): $G_t^{(n)}=\sum_{k<n}\gamma^kR_{t+k+1}+\gamma^nV(S_{t+n})$; n=1 is TD(0), n=∞ is MC. If $\|V-V^\pi\|_\infty\le\epsilon_V$ the bootstrap bias is at most $\gamma^n\epsilon_V$. No universal monotone variance ordering.
- λ-return (p193): $G^\lambda_t=(1-\lambda)\sum_{n<N}\lambda^{n-1}G^{(n)}_t+\lambda^{N-1}G_t$ for finite episodes; λ=0 is one-step, λ=1 is MC.
- Fixed-value identity (p201): $G^\lambda_t-V(S_t)=\sum_k(\gamma\lambda)^k\delta_{t+k}$ (same V in every term). Trace $e_t=\gamma\lambda e_{t-1}+\phi(S_t)$ collects the same coefficients by swapping sums (p202).
**Algorithms.**
- First-/every-visit MC (p170–171): average returns of first (or all) visits; both converge.
- Incremental MC (p172): $V\leftarrow V+\frac1{N(s)}(G-V)$; constant α gives recency weighting for non-stationary policies.
- TD(0) (p182): $V(S)\leftarrow V(S)+\alpha[R+\gamma V(S')-V(S)]$, online, per step.
- n-step TD (p190): update toward $G_t^{(n)}$.
- TD(λ) with traces (p194–199): $\delta$ computed each step, $e(S)\mathrel{+}=1$, $V(s)\leftarrow V(s)+\alpha\delta e(s)$ for all s, $e\leftarrow\gamma\lambda e$. Linear form $\theta\leftarrow\theta+\alpha\delta_te_t$ (p196).
- Forward vs backward view (p195, p200): equal offline with values held fixed; online TD(λ) changes values mid-episode and does not exactly match. True online TD(λ) uses modified traces.
**Worked examples.**
- Blackjack (p176–177): policy "hit if sum<20", state (player sum, dealer card, usable ace), 500k episodes.
- Random walk (p187–188): N=19 interior states, γ=1, rewards 0 left / +1 right, true V(i)=i/20.
**Caveats the deck stresses.**
- MC needs episodes to end (finite horizon or absorbing states) (p169).
- Bootstrap error versus variance trade-off is problem-dependent (p184, p192).

## 6. Control and off policy learning (p203–p238)
**Core idea.** Learn action values Q, improve the policy greedily while keeping exploration. Off-policy methods learn about one policy from data of another, which needs support and correction by importance sampling.
**Definitions and results.**
- Action value (p205): $Q^\pi(s,a)=E_\pi[G_t|S_t=s,A_t=a]$; greedy improvement $\pi'(s)=\arg\max_aQ^\pi(s,a)$ is at least as good (p206).
- ε-greedy (p207): $1-\epsilon+\epsilon/m$ on the greedy action, $\epsilon/m$ otherwise.
- GLIE (p209): $N_t(s,a)\to\infty$ and the policy becomes greedy in the limit. A decreasing ε alone does not guarantee visits to every state.
- MC control (p208, p210): on-policy, $\epsilon$-greedy episodes then improve. Needs its own convergence argument (variant, episodic assumptions, visit conditions, step sizes); exact improvement does not prove convergence of sampled control.
- SARSA convergence (p212–213): finite tabular, every pair visited infinitely often, GLIE, $\sum\alpha_n=\infty$, $\sum\alpha_n^2<\infty$ indexed by visits to that pair (not time), bounded rewards. Small constant α tracks change but is not this setting.
- Q-learning convergence (p220–222): w.p.1 to $Q^*$ if all pairs are visited infinitely often and Robbins–Monro step sizes per pair (e.g. $1/N_t(s,a)$). Proof: $(TQ)(s,a)=E[R+\gamma\max_{a'}Q(S',a')]$ is a γ-contraction in sup norm with unique fixed point $Q^*$; noisy updates are stochastic approximation (ODE method). Constant α converges only to a vicinity.
- Maximization bias (p225, p227): $E[\max_a\hat Q]\ge\max_aE[\hat Q]$, so Q-learning is optimistic. Double Q: select with $Q_1$, evaluate with $Q_2$. Can underestimate; with correlated neural critics not generally unbiased.
- Importance sampling (p229, p231–236): $E_\pi[X]=E_\mu[X\rho]$, $\rho=\pi/\mu$, valid only if π is absolutely continuous w.r.t. behavior (zero behavior probability cannot be repaired by a large weight). Trajectory weight $W=\prod\pi(A_t|S_t)/\beta(A_t|S_t)$; dynamics cancel; $E_\beta[WG_0]=E_\pi[G_0]$. Variance $=E[W^2G^2]-J(\pi)^2$, needs a finite second moment. Example: target picks A at all H steps, behavior with prob p, reward 1 only if all A: $X=p^{-H}1\{\text{all A}\}$, Var $=p^{-H}-1$. Exponential variance is possible, not automatic.
- Control variate (p238): $X_c=X-c(U-\mu_U)$ keeps the mean; $c^\star=\mathrm{Cov}(X,U)/\mathrm{Var}(U)$. A baseline needs correct centering.
**Algorithms.**
- SARSA (p211, p214): $Q(S,A)\leftarrow Q+\alpha[R+\gamma Q(S',A')-Q]$; on-policy.
- Q-learning (p218–219): target $R+\gamma\max_{a'}Q(S',a')$; off-policy, no importance sampling in the original algorithm.
- Expected SARSA (p223–224): target $R+\gamma E_\pi[Q(S',A)]$; averages over actions, lower variance than max; can be off-policy with π the target.
- Double Q-learning (p226).
- Diagnostics for weighted estimates (p237): inspect weight distribution and max contribution, compare across datasets, record behavior probabilities, check bias from clipping or self-normalization.
**Worked examples.**
- Cliff walking (p215–216): 4×12 grid, reward −1 per step, cliff −100; SARSA learns the safe path, Q-learning the optimal but risky edge path.
**Caveats the deck stresses.**
- On- vs off-policy (p228): safety versus optimality; Q-learning overestimates early.
- Why off-policy (p230): reuse old data, learn from expert demonstrations or logs without unsafe execution, parallel collection.
- Clipping and self-normalization change bias, not only variance (p229).

## 7. Function approximation and deep value learning (p239–p281)
**Core idea.** Large or continuous state spaces need $\hat V(s;\theta)$. Combining approximation, bootstrapping and off-policy data can diverge. DQN-style tricks (replay, target network) help in practice without general convergence proofs.
**Definitions and results.**
- Challenges (p242): non-stationary targets, bootstrapping instability, state distribution differs from the policy's.
- SGD on value error (p243–244): $\theta\leftarrow\theta+\alpha(\hat V_t-\hat V(S_t;\theta))\nabla_\theta\hat V$. Linear: $\nabla=\phi(s)$ (p245, p247).
- MC regression (p248): unbiased return G gives an SGD step on the weighted squared error; targets unbiased even if fitted parameters are biased; nonlinear optimization need not find a global minimum.
- Semi-gradient TD (p249): $\theta\leftarrow\theta+\alpha(R+\gamma\hat V(S';\theta)-\hat V(S;\theta))\nabla\hat V(S;\theta)$; the bootstrap term is held constant.
- Linear TD fixed point (p246): $A\theta=b$, $A=\Phi^\top D(I-\gamma P^\pi)\Phi$, $b=\Phi^\top Dr^\pi$, equivalent to $\Phi\theta=\Pi_DT^\pi\Phi\theta$. It differs from the least-squares projection of $v^\pi$. Convergence needs on-policy sampling D, adequate features, and stochastic approximation conditions.
- Deadly triad (p255–256): function approximation + bootstrapping + off-policy can diverge. Presence does not imply divergence, and removing one is not a general convergence theorem.
- Assumptions (p260, p262–263, p267): realizability ($Q^\star\in F$), Bellman completeness ($T^\star F\subseteq F$, $(T^\star f)(s,a)=r+\gamma E\max_{a'}f$; use $T^\pi$ for evaluation), data coverage, exploration. Separate: representing the answer does not mean a fitted update finds it; lower bounds exist despite expressive classes.
- Linear MDP (p264): $r_h=\phi^\top\theta_h$, $P_h(s'|s,a)=\sum_i\phi_i\mu_{h,i}(s')$; norm and boundedness are part of guarantees. Bellman rank (p265) unifies structures that admit polynomial learning.
- Model zoo (p266): tabular, distribution, sample, expectation, linear, learned neural.
- Four requirements (p267): expressiveness, estimation, exploration, computation.
- Distributional RL (p279): $Z^\pi(s,a)\overset d=R+\gamma Z^\pi(S',A')$, mean is $Q^\pi$; return variability differs from learned-estimate uncertainty; supports a separate risk objective.
**Algorithms.**
- Q-learning with approximation (p250): semi-gradient with max target; features $\phi(s,a)$.
- Tile coding (p252–254): M offset tilings, each state activates one cell per tiling, $\hat v=\sum_m\theta_{m,j_m(s)}$; local generalization; coarse coding scales exponentially with dimension.
- LSTD (p258–259): $\hat A\theta=\hat b$, $\hat A=\sum\phi_t(\phi_t-\gamma m_t\phi_{t+1})^\top$, $\hat b=\sum\phi_tR_{t+1}$; closed form, sample-efficient, matrix inversion; needs nonsingular $\hat A$ or explicit regularization. Not the gradient of squared TD error.
- LSVI (p261): backward ridge regression $\min\sum[\phi^\top\theta-r-V_{h+1}(s')]^2+\lambda\|\theta\|^2$ from $V_{H+1}=0$; optimistic versions add a bonus; rates depend on model, closure, data collection.
- Stability fixes (p257): gradient TD (GTD, GTD2), target networks, replay, importance weights.
- Experience replay (p268, p271): buffer of (s,a,r,s'), random mini-batches decorrelate data and reuse it; assumes off-policy data.
- DQN (p269–270, p272–273): $\theta\leftarrow\theta+\alpha(r+\gamma\max_aQ(s',a;\theta^-)-Q(s,a;\theta))\nabla Q$, target $\theta^-$ frozen and updated every C steps; double DQN against overestimation; human-level on Atari. Replay and target network do not establish nonlinear off-policy convergence.
- Dueling (p275): $Q=V+A-\frac1{|A|}\sum_bA(s,b)$; no guarantee of lower variance or convergence.
- GVF and auxiliary tasks (p276–277): extra predictions with their own γ', r' share features; test by ablation at matched budget.
- PopArt (p278): $v=\sigma[w^\top h+b]+\mu$; when (μ,σ)→(μ',σ'), $w'=\sigma w/\sigma'$, $b'=(\sigma b+\mu-\mu')/\sigma'$ keeps outputs unchanged; train on normalized targets.
- C51 (p280): fixed atoms, learn probabilities, project the Bellman-shifted distribution, KL loss.
- Quantile regression (p281): learn N quantiles, loss $L_\tau(\delta)=|\tau-1(\delta<0)|\delta$; often more stable (empirical, theory differs).
**Worked examples.**
- Mountain car (p251): underpowered car must swing back to gain momentum, reward −1 per step until the goal.
**Caveats the deck stresses.**
- Naive DQN is unstable; the tricks are essential (p273).
- ReLU can cause dead neurons; normalization can interact with RL; RL is sample-inefficient (p274).
- Non-stationarity: targets move as the policy changes (p242).

## 8. Policy gradients and actor critic (p282–p309)
**Core idea.** Optimize a parameterized policy directly. The score-function identity gives gradients that need no derivative of the dynamics. Critics trade rollout noise for bias.
**Definitions and results.**
- Why policies (p283–p284): they handle continuous actions, stochastic or multimodal policies, and partial observability with memory. Value-based methods optimize a value estimate; policy-based methods evaluate or sample an explicit rule.
- Randomization under restricted observation (p285): two hidden stages share one observation, and success needs R, then L. A memoryless policy that plays R with probability p succeeds with probability $p(1-p)$. A deterministic memoryless policy never succeeds. A policy with history or a stage counter succeeds deterministically.
- Objectives (p286–p288): start value $J_0=V^{\pi_\theta}(s_0)$; discounted $J_\rho=\mathbb E_{S_0\sim\rho}V^{\pi_\theta}(S_0)$; average reward $\bar r(\theta)=\sum_s d_{stat}^{\pi_\theta}(s)\sum_a\pi_\theta(a|s)r(s,a)$ (ergodic chain). Their gradients use different visitation weights and are not generally proportional at a fixed $\gamma$.
- Score function (p289): $\mathbb E_a[\nabla\log\pi_\theta(a|s)]=0$, so $\nabla\mathbb E[f(a)]=\mathbb E[f(a)\nabla\log\pi_\theta(a|s)]$.
- Trajectory gradient (p290–p291, p296): if the dynamics do not depend on $\theta$, $\nabla J=\mathbb E[R(\tau)\sum_t\nabla\log\pi_\theta(A_t|S_t)]$. By causality, past rewards drop out: $\nabla J=\mathbb E\sum_t\gamma^tG_t\nabla\log\pi_\theta(A_t|S_t)$.
- Policy gradient theorem (p292–p293): $\nabla J=\frac1{1-\gamma}\mathbb E_{s\sim d^\pi,a\sim\pi}[\nabla\log\pi(a|s)\,Q^\pi(s,a)]$, where $d^\pi$ is the normalized discounted occupancy. It needs Q, not the derivatives of the model. It follows from unrolling $\nabla V^\pi(s)=\sum_a\nabla\pi\,Q^\pi+\gamma\sum_{s'}P^\pi(s'|s)\nabla V^\pi(s')$.
- Baselines (p297–p301):
  - Any action-independent $b(s)$ leaves the expectation unchanged, since $b(s)\nabla\sum_a\pi=0$.
  - $b=V^\pi$ gives the advantage $A^\pi=Q^\pi-V^\pi$, but it is not the variance-minimizing baseline in general. The optimal scalar baseline is $b^*=\mathbb E[R\|\nabla\log\pi\|^2]/\mathbb E[\|\nabla\log\pi\|^2]$, which equals $\mathbb E[R]$ only when the score norm is constant.
  - A running baseline $b_{k+1}=b_k+\eta_k(G_k-b_k)$ must come from earlier rollouts: reusing the current sample in its own baseline can change the expectation.
- TD error as advantage (p304, p335): with the exact $V^\pi$, $\mathbb E[\delta_t|s,a]=A^\pi(s,a)$. With critic error $e=V-V^\pi$, the bias is $\gamma\mathbb E[e(S_{t+1})|s,a]-e(s)$.
- Representations (p306–p307): softmax $\pi\propto\exp(\theta^\top\phi(s,a))$, with score $\phi(s,a)-\mathbb E_{a'}\phi(s,a')$; Gaussian $\mathcal N(\mu_\theta(s),\sigma_\theta^2(s))$, with score $\frac{a-\mu_\theta}{\sigma_\theta^2}\nabla\mu_\theta$.
- Deterministic policy gradient (p308): $\nabla J=\frac1{1-\gamma}\mathbb E_{s\sim d^{\mu_\theta}}[\nabla_\theta\mu_\theta(s)\,\nabla_aQ^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)}]$. Replay-weighted actor updates need a stated correction. A stochastic behaviour process can supply the exploration.
**Algorithms.**
- REINFORCE (p294–p295): sample an episode, compute $G_t$ backward, and ascend $\sum_t\gamma^t(G_t-b(S_t))\nabla\log\pi_\theta(A_t|S_t)$. Unbiased, but high-variance, on-policy and sample-inefficient.
- Actor-critic (p302–p303): actor $\theta\leftarrow\theta+\alpha_\theta\nabla\log\pi\,\delta_t$; critic $\phi\leftarrow\phi+\alpha_\phi\delta_t\nabla V_\phi$, with $\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$. Two-time-scale convergence needs explicit sampling, function-class and step-size assumptions.
- n-step advantage (p305): $\hat A^{(n)}_t=\sum_{k<n}\gamma^kr_{t+k}+\gamma^nV_\phi(s_{t+n})-V_\phi(s_t)$. A larger n gives less bias and more variance; $n=\infty$ is REINFORCE with a baseline; n = 3–5 often works.
- DDPG and TD3 (p309): $\nabla J\approx\mathbb E_{s\sim D}[\nabla\mu_\theta(s)\nabla_aQ_\phi(s,a)|_{a=\mu_\theta(s)}]$, with replay and target networks. TD3 adds twin critics, delayed actor updates and smoothed target actions.
**Caveats the deck stresses.**
- An approximate critic biases the actor's direction, and off-policy critic error can misdirect a deterministic actor (p303, p309).
- The convenient baseline ($V^\pi$) is not the variance-optimal one (p297, p300).

## 9. Trust regions and entropy regularized control (p310–p338)
**Core idea.** Control the size of a policy update in the geometry of distributions (KL, Fisher), and regularize with entropy or a reference policy. Know which of these are exact constraints and which are heuristics.
**Definitions and results.**
- Policy constraints (p311): a KL constraint limits a chosen average or worst-state divergence, and its direction and state weighting must be stated. A KL penalty trades return against deviation from a reference. Regularization does not establish a hard safety or performance guarantee.
- Nonconvexity (p312): the return depends on the actions and on the state distribution they induce, so a stationary point need not be a local maximum. Results for tabular and LQR policies do not transfer to neural ones. Choose the claim (stationarity, local or global optimum) before the rate.
- Vanishing signal (p313): the score-function estimator's signal-to-noise ratio can fall exponentially with the horizon H. Remedies: variance reduction, strong critics, structured exploration, temporal abstraction.
- Stationarity rate (p314): for an L-smooth objective with exact gradients and a fixed step, $\min_{k<T}\|\nabla J(\theta_k)\|^2\le2L[J_{sup}-J(\theta_0)]/T$. This is first-order stationarity only.
- Fisher metric (p315–p320):
  - $F(\theta)=\mathbb E_{s\sim d,a\sim\pi}[gg^\top]$, with $g=\nabla\log\pi$, and $\mathbb E_d\,\mathrm{KL}(\pi_\theta\|\pi_{\theta+\Delta})=\tfrac12\Delta^\top F\Delta+o(\|\Delta\|^2)$, using $\mathbb E[g]=0$ and $-\mathbb E[\nabla^2\log\pi]=F$.
  - F is PSD and can be singular, so implementations damp it.
  - Maximizing $\nabla J\cdot\delta$ subject to $\tfrac12\delta^\top F\delta\le\epsilon$ gives $\delta^*\propto F^{-1}\nabla J$: the natural gradient, usually solved by conjugate gradient.
- Tabular NPG (p321–p322): $\pi_{k+1}(a|s)\propto\pi_k(a|s)\exp(\eta Q^{\pi_k}(s,a))$. The reference is the current policy, and subtracting V leaves the update unchanged. Global rates need a stated class, initialization and value access; they do not transfer automatically to neural policies.
- Conservative policy iteration (p323–p325):
  - The mixture is $\pi_\alpha=(1-\alpha)\pi+\alpha\pi'$, with $\frac{d}{d\alpha}J(\pi_\alpha)|_{\alpha=0}=\frac1{1-\gamma}\mathbb E_{s\sim d^\pi,a\sim\pi'}A^\pi(s,a)$.
  - A positive derivative improves only for small steps; an arbitrary α with an approximate critic guarantees nothing.
  - NPG controls a metric radius in parameter space, CPI a mixture coefficient.
- Entropy-regularized control (p337): $J_\alpha=\mathbb E\sum_t\gamma^t[r+\alpha H(\pi(\cdot|S_t))]$ and $V(s)=\mathbb E_{a\sim\pi}[Q-\alpha\log\pi]$. The greedy regularized policy is $\propto e^{Q/\alpha}$, or $\pi_{ref}e^{Q/\alpha}$ with a reference.
**Algorithms.**
- TRPO (p327): maximize $\mathbb E_{\pi_{old}}[\frac{\pi_\theta}{\pi_{old}}\hat A]$ subject to $\mathbb E_{s\sim d^{\pi_{old}}}\mathrm{KL}(\pi_{old}\|\pi_\theta)\le\delta$, by a quadratic (Fisher) approximation, approximate solves and a line search. The theoretical bound and the implemented constraint rest on different assumptions.
- PPO (p326, p328–p330):
  - Objective: $L^{clip}=\mathbb E\min(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1\pm\epsilon)\hat A_t)$, with $\rho_t=\pi_\theta/\pi_{old}$. The incentive saturates beyond $1\pm\epsilon$, in the favourable direction only.
  - Clipping bounds neither the ratios nor the KL: shared parameters move unsampled actions, and several epochs drift beyond the interval. Monitor the empirical KL, the entropy and the clip fraction, and stop early.
  - Loop: collect with $\pi_{old}$ and store its log-probabilities; estimate $\hat A$ and value targets; optimize $\ell^{clip}-c_v(V_\phi-\hat V^{targ})^2+c_HH(\pi)$ for a few epochs.
- GAE (p331–p336):
  - $\hat A^\lambda_t=\sum_{\ell\ge0}(\gamma\lambda)^\ell\delta_{t+\ell}=(1-\lambda)\sum_{n\ge1}\lambda^{n-1}\hat A^{(n)}_t$: λ = 0 is the one-step residual, λ = 1 the return minus the baseline.
  - One backward pass: $\hat A_t=\delta_t+\gamma\lambda m_t\hat A_{t+1}$, with $m_t=0$ after a true terminal; a truncation needs a bootstrap continuation, and the values are held fixed within the batch.
  - Choose λ in the estimator's own setting: more bootstrapping leans on critic quality, and a widely used value guarantees nothing for a new task.
- SAC (p338): target $Y=r+\gamma m[\min_{j=1,2}Q_{\bar\phi_j}(s',a')-\alpha\log\pi_\theta(a'|s')]$, with $a'\sim\pi_\theta$. The actor minimizes $\mathbb E[\alpha\log\pi_\theta(a|s)-\min_jQ_{\phi_j}(s,a)]$, and automatic temperature tuning targets an entropy level (Haarnoja et al., 2018).
**Caveats the deck stresses.**
- PPO clipping is a local heuristic, not a trust region (p329). TRPO's bound and its implementation differ (p327).
- A theorem for one policy class (tabular, LQR) does not establish a guarantee for a neural policy (p312, p322).

## 10. Planning and learned world models (p339–p368)
**Core idea.** A model of $p(S',R|S,A)$ lets the agent reuse data and look ahead. The price is model error, which planning amplifies with the horizon.
**Definitions and results.**
- Models (p341–p344): distribution, sample or expectation models, and table-lookup models built from counts. A linear expectation model $\mathbb E[\phi(S')|s,a]=M^\top\phi(s,a)$ is enough for a linear value. In general $V(\mathbb ES')\ne\mathbb EV(S')$, and a mean model loses the transition distribution and its tails.
- Model error is amplified (p361–p362): suppose rewards lie in [0, 1], $|r_h-\hat r_h|\le\epsilon_r$ and $\|P_h-\hat P_h\|_1\le\epsilon_P$ uniformly. Then $|V_1^\pi-\hat V_1^\pi|\le H\epsilon_r+\frac{H(H-1)}2\epsilon_P$. At stage h, $\|V_h^\pi-\hat V_h^\pi\|_\infty\le\frac{(H-h)(H-h+1)}2\epsilon_P$: the transition error multiplies the remaining value range. A uniform bound is a stronger assumption than a small average training error.
- Latent world model (p363–p364): $h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1})$, $z_t\sim q_\theta(z_t|h_t,o_t)$, with a prior $p_\theta(z_t|h_t)$. It is trained by the sequential ELBO: reconstruction terms and $-\sum_t\mathbb E_q\mathrm{KL}(q_\phi(z_t|h_t,o_t)\|p_\theta(z_t|h_t))$, plus reward and continuation heads.
- Horizon trade-off (p368): $\hat G^{(K)}_t=\sum_{k<K}\gamma^k\hat r_{t+k}+\gamma^K\hat V(\hat S_{t+K})$. A longer rollout uses more learned transitions; a shorter one leans on the terminal value. Model disagreement is a diagnostic, not a calibrated interval. Evaluate the resulting policy in the real environment or under a justified test protocol.
**Algorithms.**
- Dyna and Dyna-Q (p345–p350): after each real step, one Q-learning update, then m planning updates on past (s, a) pairs with model-generated $\hat r,\hat s'$. That is m + 1 updates per real transition, worth it only when the model is accurate.
- Experience replay (p351): resampling real transitions is implicit planning, with no model error but no extrapolation either.
- MCTS (p352–p360):
  - Steps: selection by UCB, $Q(s,a)/N(s,a)+C\sqrt{\ln N(s)/N(s,a)}$; expansion; rollout; backup ($N\leftarrow N+1$, $Q\leftarrow Q+G$). The root plays its best average.
  - It works anytime. Consistency needs a stated tree policy and enough exploration. With a learned model, more search can exploit the model's errors.
  - MCTS visits a small subset of states where value iteration sweeps all of them; AlphaGo combines both.
- Dreamer (p365): learn latent dynamics from replay, imagine rollouts from inferred states, and train a critic on imagined returns and an actor on the predicted value. DreamerV3 adds normalization and balanced objectives (Hafner et al., 2025). Transfer needs suitable observation, action and reward models.
**Caveats the deck stresses.**
- Model-based methods are sample-efficient but compound errors over long horizons; model-free methods are simple but need more data (p366–p367).

## 11. Offline and hybrid reinforcement learning (p369–p386)
**Core idea.** With a fixed log from a behaviour process and no new interaction, a policy's value is identified only where the data cover the actions it takes. Fitted methods mix statistical, approximation and iteration error, and pessimism is the main defence.
**Definitions and results.**
- Approximate VI and PI (p370–p374): $V_{t+1}=\arg\min_{V\in\mathcal H}\|V-TV_t\|^2$ may not converge, and errors in evaluation and improvement propagate. Since $J(\pi')-J(\pi)=\frac1{1-\gamma}\mathbb E_s[A^\pi(s,\pi'(s))]$, the greedy policy of an exact Q improves, but the greedy policy of an overestimating $\hat Q$ can be worse: the gain can be negative.
- Coverage (p375–p376):
  - $C(\pi)=\sup_{h,s,a:\,d_h^\pi>0}d_h^\pi(s,a)/\mu_h(s,a)$, which is infinite if the target visits what the data never show. Partial coverage can support one good comparator.
  - For action-dependent outcomes, state coverage is not enough, and an expressive policy class cannot replace missing observations.
- Inherent Bellman error (p377): $\mathcal E_{\mathcal F}=\sup_{f\in\mathcal F}\inf_{g\in\mathcal F}\|g-T^\star f\|_\mu$. It can persist with unlimited data, and it is a separate issue from coverage.
- Fitted iteration has three error sources (p378): statistical, approximation and iteration error. How they propagate depends on the horizon, the discount, the norms and the coverage.
- Unidentified counterfactual (p379–p380): logs that contain only action A, which pays 0.5 under both models, cannot tell whether the unlogged action B pays 0 or 1. The horizon alone does not imply exponential regret: a hardness result must state its information and structure.
- Conditional pessimism (p384): if $\underline V(\pi)\le V(\pi)$ for every candidate and $\hat\pi\in\arg\max\underline V$, then $V(\pi^\star)-V(\hat\pi)\le V(\pi^\star)-\underline V(\pi^\star)$. Only the comparator's pessimism error remains; establishing the lower bound is the real statistical task.
**Algorithms.**
- Fitted Q iteration (p371): regress on $Y_i=R_i+\gamma m_i\max_{a'}Q_k(S_i',a')$, holding $Q_k$ fixed. The max can query unsupported actions. Guarantees need coverage, closure and control of the regression error.
- CQL (p381): add $\alpha\mathbb E_{s\sim D}[\log\sum_ae^{Q(s,a)}-\mathbb E_{a\sim\beta}Q(s,a)]$ to the Bellman error. This pushes high values down and props up the logged actions (Kumar et al., 2020).
- IQL (p382): fit V as an upper expectile ($\tau>1/2$) of the dataset's action values and Q to $r+\gamma mV(s')$, then extract the policy by advantage-weighted behaviour cloning. It never queries unseen actions (Kostrikov et al., 2021).
- Pessimistic model-based RL (p383): plan with $\tilde r=\hat r-\lambda u(s,a)$. The penalty must track return-relevant model error (MOPO; Yu et al., 2020).
- Hybrid RL (p385–p386): refit on a controlled mixture of offline and online samples (Hy-Q, Song et al., 2023, under bounded bilinear rank). Cal-QL calibrates conservative values to a reference policy's scale before online fine-tuning (Nakamoto et al., 2023). Specify the online budget and the replay weights, and judge offline return and fine-tuning efficiency separately.
**Caveats the deck stresses.**
- Guarantees need an explicit relation between the candidate policy and the logged distribution (p376). A confident neural model can still be wrong in unsupported regions (p383).

## 12. Imitation, inverse RL and generative policies (p387–p414)
**Core idea.** Learn a policy from expert demonstrations with no reward. The central issue is the mismatch between expert-induced and learner-induced state distributions; fixes are on-policy labels, occupancy matching, max-entropy reward inference, or generative (diffusion) policies.
**Definitions and results.**
- Imitation setting (p388): demonstrations $\{\tau_m\}$ from $\pi_{expert}$, no reward; goal is $\pi_\theta$ imitating it.
- BC error under expert states (p390): $\epsilon_E=\frac1H\sum_t E_{S_t\sim d^{\pi_E}}P(\pi(S_t)\ne\pi_E(S_t))$. Training distribution is expert-induced, deployment is learner-induced.
- Finite-horizon BC bound (p391, p393): costs in $[0,1]$, a first disagreement can change the whole remaining path; $|C(\pi)-C(\pi_E)|\le\sum_t\sum_{h\le t}\epsilon_h\le H^2\bar\epsilon$. Quadratic, not exponential.
- Discounted bound (p392): $0\le c\le C_{max}$ gives $|J_c(\pi)-J_c(\pi_E)|\le C_{max}\epsilon/(1-\gamma)^2$. Turning a finite dataset into $\epsilon$ needs separate generalization assumptions.
- Learner-state error (p396, p399): error $\epsilon_\pi$ under the learner's own states and one-deviation cost-to-go increase $\le u$ give $C(\pi)-C(\pi_E)\le uH\epsilon_\pi$. If mistakes are irreversible, $u$ can grow with $H$ (recoverability sets the horizon factor).
- Occupancy bound (p402): with $d^\pi(s,a)=(1-\gamma)\sum_t\gamma^tP(S_t=s,A_t=a)$ and $|r|\le R_{max}$, $|J(\pi)-J(\pi_E)|\le\frac{R_{max}}{1-\gamma}\|d^\pi-d^{\pi_E}\|_1$. State matching alone misses action-dependent rewards (p402–p404).
- MaxEnt trajectory inference (p405–p407): maximize relative entropy subject to $E_pf(\tau)=\hat f_E$; solution $p_\theta(\tau)=p_0(\tau)e^{\theta^\top f(\tau)}/Z(\theta)$; $\theta$ acts as reward weights. Likelihood $\ell=\theta^\top\hat\mu_E-\log Z$, gradient $\hat\mu_E-E_{p_\theta}f$ (feature matching).
- Causal soft control (p408): $Q_h=r_\theta+E[V_{h+1}]$, $V_h=\log\sum_ae^{Q_h}$, $\pi_h=e^{Q_h-V_h}$. Inferred reward depends on the chosen features and entropy model.
- Diffusion/score facts (p409–p412): noise loss $E\|\epsilon-\epsilon_\theta(\tau_k,k,c)\|^2$; reverse SDE uses the score $\nabla\log p_t$; classifier-free guidance $s_{guide}=s_{uncond}+w[s_{cond}-s_{uncond}]$.
**Algorithms.**
- BC (p389): $\max_\theta E_{(s,a)\sim D_{expert}}\log\pi_\theta(a|s)$. Good only if the expert is good and test-time state visitation is similar.
- DAgger (p394–p398): roll out learner or mixture $\beta_k\pi_E+(1-\beta_k)\pi_k$, expert labels visited states, aggregate, refit. The guarantee combines online-learning regret, class error and a cost-to-go assumption. Queries are $NH$ for $N$ one-rollout iterations; an iteration guarantee is not a label-complexity guarantee.
- AggreVaTe (p400–p401): at learner roll-in states, estimate expert continuation cost per candidate action and train a cost-sensitive learner. It needs expert continuation access, stronger than an action label.
- Diffusion planning/policy (p410, p413): denoise a trajectory, execute a prefix, replan (Janner 2022); Diffusion Policy generates action chunks with receding horizon (Chi 2023).
**Caveats the deck stresses.**
- BC compounds errors off the expert's states (p393).
- State which policy a DAgger theorem concerns, a mixture or an iterate (p397).
- Occupancy estimation has its own statistical and computational cost (p403).
- Generated trajectories need dynamic-feasibility and support checks (p410).
- Reward or goal conditioning does not certify feasibility or return; evaluate in the decision environment (p412).
- Architecture alone does not fix the learning problem; imitation versus reward optimization differ by signal and coverage (p414).
**Readings named.** Janner et al. 2022; Song et al. 2021; Ho and Salimans 2022; Chi et al. 2023.

## 13. Preference learning and RLHF (p415–p435)
**Core idea.** Hand-designed rewards invite hacking and Goodhart effects (p416), so learn from pairwise comparisons. Optimize reward minus a KL penalty to a reference, or skip the reward model (DPO and relatives).
**Definitions and results.**
- Random utility / Bradley-Terry (p417–p419): $U=r+\varepsilon$; unit Gumbel noise gives $P(y_w\succ y_l)=\sigma(R(y_w)-R(y_l))$. MLE is binary cross-entropy on pairs.
- KL-regularized optimum (p420): $\max_\pi E_\pi r-\beta KL(\pi\|\pi_{ref})$ gives $\pi^\star=\pi_{ref}e^{r/\beta}/Z(x)$. The reference must have positive mass on every response the policy may consider.
- RLHF objective (p421): $J(\pi)=E_xE_{y\sim\pi}[\hat r-\beta KL(\pi\|\pi_{ref})]$. The reward model's errors stay in the problem; a penalty is not a hard constraint.
- Exact gap (p422): $J_\beta(\pi)=\beta\log Z-\beta KL(\pi\|\pi^\star)$, so $J_\beta(\pi^\star)-J_\beta(\pi)=\beta KL(\pi\|\pi^\star)$.
- Error decomposition (p423): if $|r-\hat r|\le\epsilon_r$ uniformly and $\hat\pi$ is $\epsilon_{opt}$-optimal for $\hat r$, then $J_\beta(\pi^\star;r)-J_\beta(\hat\pi;r)\le2\epsilon_r+\epsilon_{opt}$. The uniform bound depends on preference-model error and coverage.
- DPO reparameterization (p426): $r=\beta\log\frac{\pi^*}{\pi_{ref}}+\beta\log Z(s)$; $Z$ cancels in pairwise differences.
- DPO assumptions (p431): Bradley-Terry preferences and adequate expressiveness; finite offline pairs constrain only the comparisons they cover; a fixed reference does not remove reward misspecification or overoptimization.
**Algorithms.**
- PPO for RLHF (p424–p425): frozen $\pi_{old}$ rollouts, reward model plus reference KL, optional critic, ratio-clipped actor surrogate. State where KL enters and what is frozen.
- DPO (p427–p430): $z_\theta=\beta[\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}]$, $L_{DPO}=-E\log\sigma(z_\theta)$. Gradient $-\beta\sigma(-z_\theta)[\nabla\log\pi_\theta(y_w)-\nabla\log\pi_\theta(y_l)]$; misordered pairs get larger weight.
- IPO / KTO / SimPO (p433): IPO uses a squared log-ratio loss; KTO uses desirable/undesirable labels without pairs; SimPO uses length-normalized log-probability and a target margin, with no reference model.
- REBEL (p434–p435): $\pi_{t+1}\propto\pi_te^{\eta r}$, so the normalizer cancels in pairs. Loss $\min_\theta\hat E\big[\big(\frac{g_{\theta,t}(y)-g_{\theta,t}(y')}{\eta}-(r(x,y)-r(x,y'))\big)^2\big]$, with $g=\log(\pi_\theta/\pi_t)$. Needs reward differences; binary preferences give no magnitudes.
**Caveats the deck stresses.**
- Preference fit, task reward and policy KL are judged separately on held-out data (p430, p432).
- Compare losses with data construction and sampling access fixed (p432).
- DPO is supervised on fixed pairs; online reward sampling adds different information (p431).
**Readings named.** Azar et al. (IPO); Ethayarajh et al. (KTO); Meng et al. (SimPO); Gao et al. 2024 (REBEL).

## 14. Reasoning and verifiable rewards (p436–p447)
**Core idea.** Treat generation as a token-level MDP with a terminal verifier reward and optimize with group-relative policy gradients (GRPO family). Measured reward reflects only what the verifier covers.
**Definitions and results.**
- Token MDP (p437): $s_t=(x,y_{<t})$, $a_t=y_t$, deterministic transitions; one-response bandit and token-level updates differ in ratios and normalizations.
- Verifier reward (p438): $r=1\{verify(x,y)=pass\}$ (math answer match, code tests, formal checker). A pass measures the verifier's criterion, not general correctness. Record parse failures, timeouts and invalid outputs.
- Baseline coupling (p441): without standardization $E[(r_i-\bar r)g_i]=(1-\frac1G)\nabla E[r_i]$. A leave-one-out baseline is independent of $y_i$, and $r_i-b_{-i}=\frac G{G-1}(r_i-\bar r)$. Dividing by the group sd adds another dependence.
**Algorithms.**
- Group advantage (p439): $\hat A_i=(r_i-\bar r)/(s_r+\varepsilon)$ over $G$ samples from $\pi_{old}$, treated as fixed in the update (DeepSeekMath 2024).
- GRPO surrogate (p440): $\frac1G\sum_i\frac1{T_i}\sum_t\min\{\rho_{i,t}\hat A_i,\mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\}$, $\rho=\pi_\theta/\pi_{old}$ per token. A reference KL can be added; length averaging and group standardization are consequential.
- DAPO (p443): decoupled asymmetric clipping, dynamic sampling, token-level loss aggregation, overlong reward shaping (Yu 2025). Credit the whole setup unless an ablation isolates a part.
- Dr GRPO (p444): length normalization and group-sd normalization bias token and prompt weights; it modifies the normalization (Liu 2025).
- GGPO (p447): stale behavior policies in verifiable-reward training; log-mean-exponential contrast of KL-adjusted rewards, no learner-to-behavior importance weights; theory assumes coverage and finite-data conditions (Zhang and Zhang 2026 preprint).
**Caveats the deck stresses.**
- Audit labels and answer extraction on a held-out sample; keep training rewards apart from final tests; inspect high-reward failures for exploitation (p445).
- Higher reward can mean better task performance, better score exploitation, or both (p445).
- Report prompts, rollouts, tokens, steps, limits; separate pass@1 from best-of-K or pass@K; match inference budgets; count verifier and sampling cost (p446).
- R1-Zero, R1 and distillation differ in pretraining, data, reward design and compute (p442).
**Readings named.** Shao et al. 2024 (DeepSeekMath); DeepSeek R1 (rev. Jan 2026); Yu et al. 2025 (DAPO); Liu et al. 2025 (Understanding R1-Zero-Like Training); Zhang and Zhang 2026 (GGPO).

## 15. Multi turn agents and value based post training (p448–p456)
**Core idea.** In multi-turn settings an action changes future context, so terminal success must be credited across earlier decisions via future-return values or relative-future-reward regression.
**Definitions and results.**
- Multi-turn state (p449): $S_h$ = history, tool outputs, task state. Fixed historical final turns omit policy-induced context shift.
- Credit (p450): $Q^\pi_h(s,a)=E_\pi[\sum_{t\ge h}r_t|s,a]$, $A=Q-V$. Immediate quality and future value can differ; estimates depend on the continuation policy and environment.
- Q-sharp (p453–p454): deterministic finite-horizon MDP, $V^\star_h(s)=\beta\log E_{a\sim\pi_{ref}}e^{Q^\star_h/\beta}$, $\pi^\star_h\propto\pi_{ref}e^{Q^\star_h/\beta}$. With $G_h$ the remaining reward under the reference, $Q^\star_h=\beta\log E_{\pi_{ref}}[e^{G_h/\beta}|s,a]$. It learns the reference return distribution on aggregated data, and the reference weights stay fixed (Zhou 2025).
**Algorithms.**
- REFUEL (p451–p452): at an on-policy history sample two actions, finish both under $\pi_t$, terminal scores $R,R'$. Loss $\min\hat E\big[\big(\frac1\eta(\log\frac{\pi_\theta(a|s_h)}{\pi_t(a|s_h)}-\log\frac{\pi_\theta(a'|s_h)}{\pi_t(a'|s_h)})-(R-R')\big)^2\big]$. It is DAgger-like (learner-induced histories); guarantees need coverage and approximation assumptions (Gao 2025).
- SCALE (p456): adds probability of eventual success to return prediction and uses its changes for step credit, modulated by reliability. Open question: which ablations separate better credit from extra supervision (Liu 2026 preprint).
**Caveats the deck stresses.**
- Evaluate tool actions at environment level: persistence of effects, latency, cost and resource limits in state or objective, observation versus execution errors, final success beyond call validity (p455).
- A critic can fit aggregate returns yet miss local action differences (p456).
**Readings named.** Gao et al. 2025 (REFUEL); Zhou et al. 2025 (Q sharp); Liu et al. 2026 (SCALE).

## 16. Multi agent reinforcement learning (p457–p462)
**Core idea.** With several learning agents the environment is non-stationary from each agent's view. Training can be centralized but execution must use only local information. Team return is not an equilibrium notion.
**Definitions and results.**
- Markov game (p458): joint action $a=(a_1..a_N)$, $S_{t+1}\sim P(\cdot|S_t,A_t)$, rewards $r_i$. Cooperative agents share an objective, others do not; other agents' learning changes the effective environment; local observations make it partially observed.
- Nash equilibrium (p462): $V_i(\pi_i^\star,\pi_{-i}^\star)\ge V_i(\pi_i,\pi^\star_{-i})$ for all $i,\pi_i$. Finite strategic-form games admit mixed equilibria; Markov-game statements need their own horizon and policy assumptions. High team return alone does not establish equilibrium.
- QMIX (p460): $Q_{tot}=g_s(Q_1,\dots,Q_N)$ with $\partial Q_{tot}/\partial Q_i\ge0$ via state-conditioned hypernetwork weights. Locally greedy actions are then jointly greedy for the represented $Q_{tot}$, but the true joint value need not lie in this class (Rashid 2018).
**Algorithms.**
- CTDE (p459): central critics may use global state or joint actions in training; execution uses own observation or history. Training-only information must not leak into execution.
- Graph communication (p461): $m_i=\mathrm{AGG}_{j\in N(i)}\psi(h_i,h_j,e_{ij})$, $h_i'=\phi(h_i,m_i)$. Shares parameters across team sizes; gives no coordination guarantee.
**Caveats the deck stresses.**
- Compare baselines under the same communication constraints (p459).
- Delays, missing messages and bandwidth belong in evaluation (p461).
- Test held-out partners, opponents and communication failures (p462).
**Readings named.** Rashid et al. 2018 (QMIX).

## 17. Linear quadratic control (p463–p478)
**Core idea.** Linear dynamics, a quadratic cost, $R\succ0$ and a known model make the value quadratic and the optimal policy a linear feedback, computed by a Riccati recursion. Policy gradients on LQR have a clean structure.
**Definitions and results.**
- Finite-horizon LQR (p464–p467):
  - Setting: $x_{t+1}=Ax_t+Bu_t+w_t$, $\ell=x^\top Qx+u^\top Ru$, with $Q,Q_H\succeq0$ and $R\succ0$.
  - With the ansatz $V_t(x)=x^\top P_tx+c_t$, the backup is quadratic in (x, u), and the noise adds only $\mathrm{tr}(P_{t+1}W_t)$, which does not depend on u.
  - With $S_t=R+B^\top P_{t+1}B\succ0$: $u_t^\star=-K_tx$, where $K_t=S_t^{-1}B^\top P_{t+1}A$.
- Riccati recursion (p468–p470): $P_t=Q+A^\top P_{t+1}A-A^\top P_{t+1}B(R+B^\top P_{t+1}B)^{-1}B^\top P_{t+1}A$ and $c_t=c_{t+1}+\mathrm{tr}(P_{t+1}\Sigma_w)$, solved backward. The gains are linear and time-varying; the solution is exact.
- Infinite horizon (p471): if $(A,B)$ is stabilizable and $(A,Q^{1/2})$ detectable, the algebraic Riccati equation gives a K with $\rho(A-BK)<1$. With persistent noise, use an average or discounted cost.
- Why it is tractable (p472): linear dynamics, quadratic cost, $R\succ0$, a known model and stability conditions; each assumption has a role.
- Policy gradient (p474; Fazel et al., 2018): for $C(K)=\mathbb E\sum_t\ell(x_t,-Kx_t)$, $\nabla_KC=2E_K\Sigma_K$, where $\Sigma_K=\mathbb E\sum_tx_tx_t^\top$ is the state visitation and $E_K=(R+B^\top P_KB)K-B^\top P_KA$ the improvement residual.
**Algorithms.**
- Policy iteration (p473): from a stabilizing K, evaluate $P_K=Q+K^\top RK+(A-BK)^\top P_K(A-BK)$ (a Lyapunov equation), then improve to $K_{new}=(R+B^\top P_KB)^{-1}B^\top P_KA$.
- Natural and Gauss–Newton steps (p475–p476): $K-\eta\nabla C\,\Sigma_K^{-1}=K-2\eta E_K$ removes the visitation factor, and $K-\tfrac12S_K^{-1}\nabla C\,\Sigma_K^{-1}=S_K^{-1}B^\top P_KA$ is exact policy improvement. Steps must keep the closed loop stable.
- Convex synthesis (p477–p478): an LMI in $X\succ0$ and $Y=KX$ certifies stability, with $K=YX^{-1}$. System level synthesis optimizes the closed-loop responses $\Phi_x,\Phi_u$, subject to $(zI-A)\Phi_x-B\Phi_u=I$, under an $H_2$ cost and convex constraints (locality, robustness); then $K=\Phi_u\Phi_x^{-1}$ (Wang, Matni and Doyle).
**Caveats the deck stresses.**
- A stabilizing initialization is part of the policy-iteration setting (p473). A full LQR cost SDP needs more than the stability LMI (p477).

## 18. Sequential decisions in quantitative finance (p479–p488)
**Core idea.** A trading problem is an MDP only under a careful contract on information, state, costs and counterfactuals. A backtest is conditional on its execution and market assumptions.
**Definitions and results.**
- Execution MDP (p480): $S_t=(q_t,\text{time left},\text{market observations})$, $A_t=v_t$, $q_{t+1}=q_t-v_t$. The reward is minus the execution cost, with a terminal inventory penalty; participation and inventory limits are action constraints.
- Counterfactuals (p481): a logged fill records only the action submitted. Another action could have filled differently or moved the price. A simulator supplies counterfactuals through assumptions, and the backtest is conditional on them. Exercise: list which parts of the next state replay from data and which need a counterfactual model.
- Portfolio MDP (p482): choose post-decision weights $w_t$ from $\mathcal F_t$, before $R_{t+1}$, with $W_{t+1}/W_t=1+w_t^\top R_{t+1}-c_t$ and $c_t=c(w_t,\tilde w_t,S_t)$. Here $\tilde w_t$ is the pre-trade portfolio after the previous returns. Constraints: budget, position bounds, leverage. **The state includes the holdings** and every variable the cost model needs.
- Reward and mandate (p483): one choice is $\mathbb E\sum_{t<H}[\log(1+w_t^\top R_{t+1}-c_t)-\lambda w_t^\top\hat\Sigma_tw_t]$. The wealth multiplier must stay positive; $\hat\Sigma_t$ uses $\mathcal F_t$ only; the quadratic penalty is a chosen surrogate; hard leverage and liquidity limits belong in the feasible set.
- Lower-tail CVaR (p484): $\mathrm{CVaR}_\alpha^{lower}(Z)=\frac1\alpha\int_0^\alpha F_Z^{-1}(u)\,du=\sup_\eta\{\eta-\frac1\alpha\mathbb E[(\eta-Z)_+]\}$. This form handles atoms. State the reward or loss convention and the meaning of α.
- Dynamic risk (p485): a risk functional ranks return distributions. Static CVaR of the total return need not satisfy the ordinary Bellman recursion, so a time-consistent dynamic risk model or an augmented state may be needed. Ask whether the objective preserves the Markov structure the algorithm relies on.
- Timing contract (p486): date every feature, forecast, decision, fill, cost and outcome. Split chronologically into training, selection and final evaluation. Fit normalization and feature selection on the permitted window only. Include spreads, fees, financing and impact. Report how the behaviour data support the proposed policy's actions.
- Worked example (p487): liquidate 10 units over two dates at $C(v_0)=v_0^2+v_1^2+q_1^2=v_0^2+2(10-v_0)^2$. Then $C'=6v_0-40=0$, so $(v_0,v_1)=(20/3,10/3)$. Trading equally costs 75; the optimum costs 200/3.
- Research exercise (p488): verify the optimizer. Add stochastic prices and an impact or fill model. Compare uniform, model-based and learned execution, holding the training interactions and evaluation paths fixed. Report costs, remaining inventory, uncertainty and constraint violations, and explain which assumptions identify the counterfactual costs.

## 19. Evaluation, reproducibility and deployment (p489–p496)
**Core idea.** An RL result varies over training runs and over evaluation episodes. A credible comparison fixes information, budgets and the selection procedure before testing.
**Definitions and results.**
- Uncertainty (p490): separate training randomness from evaluation noise, and use several independent runs. Give the uncertainty of aggregates as well as of single tasks; robust aggregates and performance profiles show different things (Agarwal et al., 2021).
- Matched comparisons (p491): fix data access, interactions and tasks, and record tuning effort and selection rules. For LLMs, report tokens and samples; for world models, include model and planning compute; for discovered algorithms, separate the cost of discovery from the use of the frozen rule.
- Off-policy evaluation (p492–p493):
  - Three estimators: importance sampling (needs support), a direct model, and doubly robust, $\hat V_{DR}=\hat V_0(S_0)+\sum_t\gamma^tW_{0:t}[R_{t+1}+\gamma\hat V_{t+1}(S_{t+1})-\hat Q_t(S_t,A_t)]$ with $W_{0:t}=\prod_{k\le t}\pi/\beta$.
  - Doubly robust is unbiased with correct weights for fixed fitted functions, and an accurate model lowers its variance.
  - Fit on independent data or justify the sample split, and diagnose weight concentration and model error.
- Chronological evaluation (p494): fit on the training interval, select on a later validation interval, and freeze the selection procedure before the test interval; repeat in a rolling or expanding window. Overlapping reward windows leak across a boundary, so set an exclusion interval from the label horizon and the information timing. Repeated strategy search changes how a final estimate should be read.
- Deployment (p495): monitor state and action distributions against the training support; track realized cost, outcomes and constraint violations; re-evaluate a reward proxy when behaviour changes; define fallbacks for missing data and failed executions; version the policy, data, environment and scoring.
- Reading (p496): DeliveryGym (Kang et al., 2026) evaluates whole courier shifts under persistent resource constraints, and keeps its adaptive curriculum separate from a fixed evaluation suite.

## 20. Algorithm discovery and research frontiers (p497–p501)
**Core idea.** The update rule itself can be learned: inner learning updates the agent, outer learning changes how it learns (p498).
**Definitions and results.**
- Learned rule $\theta_{k+1}=U_\phi(\theta_k,D_k)$; outer objective $\max_\phi \mathbb{E}_{M\sim p_{train}}[J_M(\theta_K(\phi))]$ (p498).
- DiscoRL (Oh et al. 2025): a meta network defines losses for the policy and extra predictions; a population of agents supplies experience; meta-gradients optimize the shared rule; the rule then trains new agents on held-out environments (p499).
- Seminar projects: group-centered vs leave-one-out baselines at fixed rollout cost; model error vs best planning horizon; offline improvement as action support shrinks; mean vs lower-tail reward; audit of long-horizon benchmarks for hidden information or reward shortcuts (p501).
**Caveats the deck stresses.**
- Transfer claims need held-out problems: separate discovery and transfer environments, match training budgets, report discovery compute apart from reuse compute, test changes in observations, actions, rewards, horizon (p500).
- Open question: does the rule transfer because it captures structure, or because the tasks resemble its discovery distribution (p500)?

## 21. Mathematical appendix and extended proofs (p502–p573)

**Concentration inequalities**
- Sub-Gaussian: centered $X$ is $\sigma$-sub-Gaussian if $\mathbb{E}e^{\lambda X}\le e^{\lambda^2\sigma^2/2}\ \forall\lambda$; Chernoff gives $P(X\ge u)\le e^{-u^2/(2\sigma^2)}$, two-sided with factor 2 (p503, p505). $\sigma$ is a tail scale, not the sd. Bounded in $[a,b]$: $\sigma=(b-a)/2$; Gaussian; Rademacher $\sigma=1$ (p504). With $u=\sigma\sqrt{\ln T}$ the tail is $O(1/T^2)$, so a union bound over $T$ events works; this underlies UCB (p505).
- Hoeffding's lemma: $X\in[a,b]\Rightarrow \mathbb{E}e^{\lambda(X-\mathbb{E}X)}\le \exp(\lambda^2(b-a)^2/8)$, no independence needed (p506). Proof: $h(\lambda)=\log\mathbb{E}e^{\lambda(X-\mathbb{E}X)}$, $h(0)=h'(0)=0$, $h''(\lambda)$ is the tilted variance (p507); tilted law stays in $[a,b]$ so $h''\le (b-a)^2/4$ (p508); integrate twice, $h(\lambda)\le\lambda^2(b-a)^2/8$ (p509).
- Hoeffding inequality: i.i.d. $X_i\in[0,1]$, $P(\bar X-\mathbb{E}X\ge u)\le e^{-2nu^2}$ (p510); two-sided $2e^{-2nu^2}$, general range $2n(u/(b-a))^2$ (p514). Proof: Markov on $e^{\lambda(\cdot)}$, independence factors the MGF, apply the lemma, optimize $\lambda^*=4nu$ (p511–p513). Confidence radius $u=\sqrt{\ln(2/\delta)/(2n)}$ (p514).
- Hoeffding–Azuma: $\mathbb{E}[D_t|\mathcal F_{t-1}]=0$, $|D_t|\le c_t$: $P(\sum D_t\ge x)\le\exp(-x^2/(2\sum c_t^2))$ (p516).
- Bernstein: i.i.d. in $[a,b]$, variance $\sigma^2$: $P(\frac1n\sum X_i-\mathbb{E}X\ge\epsilon)\le\exp(-n\epsilon^2/(2(\sigma^2+\frac13(b-a)\epsilon)))$; tighter when variance is small (p517).
- Freedman: martingale, $|D_t|\le b$, predictable variation $W_n=\sum\mathbb{E}[D_t^2|\mathcal F_{t-1}]$: $P(M_n\ge x,W_n\le v)\le\exp(-x^2/(2(v+bx/3)))$ (p518).
- Self-normalized bound (Abbasi-Yadkori 2011): $S_t=\sum x_i\eta_i$, $V_t=\lambda I+\sum x_ix_i^\top$; w.p. $1-\delta$ for all $t$, $\|S_t\|_{V_t^{-1}}\le R\sqrt{2\log(\det(V_t)^{1/2}/(\det(\lambda I)^{1/2}\delta))}$ (p519). Ridge ellipsoid, $\|\theta^\star\|\le S$: $\|\hat\theta_t-\theta^\star\|_{V_t}\le\beta_t=R\sqrt{2\log(\cdot)}+\sqrt\lambda S$, hence $|x^\top(\hat\theta_t-\theta^\star)|\le\beta_t\sqrt{x^\top V_t^{-1}x}$ (p515, p520).
- Choice by assumption: Hoeffding (independent bounded), Bernstein (plus variance), Azuma (bounded martingale), Freedman (plus predictable variance), self-normalized (predictable features, sub-Gaussian noise) (p521). Union bound template: $\delta_i=\delta/|E|$ for finite events, covering arguments otherwise (p522).

**Bandit regret proofs**
- Greedy has linear regret: with probability $p>0$ it locks onto a gap-$\Delta$ arm after $t_0$, so $\mathbb{E}R_T\ge p\Delta(T-t_0)=\Omega(T)$ (p523–p524). Worst-instance failure, not every instance.
- Epsilon-greedy: $\mathbb{E}R_T=\sum_t[\epsilon_t\bar\Delta+(1-\epsilon_t)\mathbb{E}\Delta_{\hat a_t}]$, $\bar\Delta=K^{-1}\sum_a\Delta_a$ (p525). Uniform concentration: $c_n=\sqrt{\log(2KT/\delta)/(2n)}$, union over arms and $n\le T$ (p526). If every arm has $n_0$ samples and $c_{n_0}<\Delta_{\min}/2$ the best arm is identified; $n_0=O(\Delta_{\min}^{-2}\log(KT/\delta))$ (p527). Fixed $\epsilon$ pays linear cost; log-regret needs a decaying schedule and a gap lower bound (Auer et al. 2002) (p528). A final-estimation-error bound or an expected-count bound is not a regret or minimax proof (p529).
- Lai–Robbins lower bound (Bernoulli, uniformly efficient policy): change only suboptimal arm $a$ to $q>p^\star$ (p530); transcript KL $=\mathbb{E}_P N_a(T)\,kl(p_a,q)$ (p531); test event $A_T=\{N_a<T/2\}$, data processing $KL\ge kl(P(A_T),Q(A_T))$ (p532); gives $\liminf\mathbb{E}N_a/\log T\ge 1/kl(p_a,p^\star)$ after $\alpha\downarrow0,q\downarrow p^\star$ (p533); with $\mathbb{E}R_T=\sum\Delta_a\mathbb{E}N_a$, $\liminf \mathbb{E}R_T/\log T\ge\sum_{a:\Delta_a>0}\Delta_a/kl(p_a,p^\star)$ (p534).
- Minimax: $K$ instances, arm $i$ raised from $1/2$ to $1/2+\delta$, each pull gives $O(\delta^2)$ KL, change of measure; $\delta\asymp\sqrt{K/T}$ gives a worst-case lower bound of order $\sqrt{KT}$ (p535).
- UCB1 (rewards in $[0,1]$, $U_a=\hat\mu_a+\sqrt{2\log t/N_a}$): $\mathbb{E}R_T=\sum\Delta_a\mathbb{E}N_a$, threshold $\ell_a=\lceil 8\log T/\Delta_a^2\rceil$ (p536). Selecting $a$ implies $\Delta_a\le 2c_t(N_a)$ on the good event (p537); beyond $\ell_a$ a bad event must occur (p538–p539). Hoeffding gives $t^{-4}$ per count, union $\le 2t^{-2}$, summable to $\pi^2/3$ (p540–p541). So $\mathbb{E}N_a\le 8\log T/\Delta_a^2+1+\pi^2/3$ (p542), $\mathbb{E}R_T\le 8\sum\log T/\Delta_a+C\sum\Delta_a$ (p543). Gap-free: split at $\eta$, $\mathbb{E}R_T\le T\eta+CK\log T/\eta$, $\eta\asymp\sqrt{K\log T/T}$ gives $O(\sqrt{KT\log T}+K)$ (p544).
- LinUCB: concentration, optimism, information gain via the elliptical potential lemma, regret (p545). Lemma: $V_t=V_{t-1}+x_tx_t^\top$, $\|x_t\|\le L$, $u_t=x_t^\top V_{t-1}^{-1}x_t$: $\sum\min\{1,u_t\}\le2\log(\det V_T/\det V_0)\le 2d\log(1+TL^2/(\lambda d))$ (p546). Steps: rank-one determinant lemma $\det V_t=\det V_{t-1}(1+u_t)$ (p547), telescoping log-det (p548), $\min\{1,u\}\le2\log(1+u)$ (p549), AM-GM on eigenvalues/trace (p550); with $\lambda\ge L^2$ and Cauchy–Schwarz $\sum\|x_t\|_{V_{t-1}^{-1}}\le\sqrt{2Td\log(1+TL^2/(\lambda d))}$ (p551). On the confidence event $\Delta_t\le2\beta_{t-1}\|x_t\|_{V_{t-1}^{-1}}$, so $R_T\le2\beta_T\sqrt{2Td\log(1+TL^2/(\lambda d))}$ (p552); expected regret also needs the complement event.
- Explore-then-commit ($m$ pulls per arm, $Km\le T$): $\mathbb{E}R_T=m\sum\Delta_a+(T-Km)\mathbb{E}\Delta_{\hat a}$ (p553). Wrong commit needs an error $\ge\Delta_a/2$ on one of two arms (p554); Hoeffding: $P(\hat a=a)\le2e^{-m\Delta_a^2/2}$ (p555). Gap-free: $c=\sqrt{\log(2KT)/(2m)}$, $\Delta_{\hat a}\le2c$ w.p. $1-1/T$ (p556); $\mathbb{E}R_T\le Km+2T\sqrt{\log(2KT)/(2m)}+1$ (p557); $m\asymp(T^2\log(KT)/K^2)^{1/3}$ gives $\tilde O(K^{1/3}T^{2/3})$ (p558).

**Policy-gradient score estimator**
- $p_\theta(\tau)=p(s_0)\prod\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)$; transition and initial terms do not depend on $\theta$, so $\nabla\log p_\theta(\tau)=\sum_t\nabla\log\pi_\theta(a_t|s_t)$ (p559–p560). Discounted reward-to-go $\sum_{k\ge t}\gamma^kR_{k+1}=\gamma^tG_t$ (p561). Estimator $\hat g(\tau)=\sum_t\gamma^t\nabla\log\pi(A_t|S_t)(G_t-b(S_t))$, $\mathbb{E}\hat g=\nabla J$ under on-policy assumptions; bootstrapped targets can bias it; off-policy needs a separate correction (p562).

**Simulation lemma**
- Same policy in $(r,P)$ and $(\hat r,\hat P)$, rewards in $[0,1]$, $|r_h-\hat r_h|\le\epsilon_r$, $\|P_h-\hat P_h\|_1\le\epsilon_P$ (p563–p564). Subtract Bellman equations, add/subtract $P^\pi_h\hat V^\pi_{h+1}$ (reward, propagated, transition error) (p565); $\Delta_h\le\epsilon_r+\Delta_{h+1}+(H-h)\epsilon_P$ using the sup-norm nonexpansive kernel and $\hat V\le H-h$ (p566); unrolled, $\Delta_1\le H\epsilon_r+\frac{H(H-1)}2\epsilon_P$ (undiscounted) (p567). Policy selection: if $|V^\pi-\hat V^\pi|\le B$ for all candidates and $\hat\pi$ is $\epsilon_{opt}$-optimal in the model, $V^{\pi^\star}-V^{\hat\pi}\le2B+\epsilon_{opt}$; uniformity matters when selection uses the data (p568).

**KL-regularized objective and DPO derivation**
- Per prompt $x$: $\max_\pi\sum_y\pi r-\beta\sum_y\pi\log(\pi/\pi_{ref})$ s.t. simplex; concave, separable per prompt without parameter sharing; a neural policy couples prompts (p569). Lagrangian with multiplier $\lambda$ (p570); KKT: $r-\beta\log(\pi/\pi_{ref})-\beta-\lambda=0$, so $\pi^\ast\propto\pi_{ref}\exp(r/\beta)$ (p571); normalized $\pi^\ast=\pi_{ref}\exp(r/\beta)/Z(s)$, $Z=\sum_a\pi_{ref}\exp(r/\beta)$; this closed form is the basis for DPO (p572). The slides stop at the optimal policy; the DPO loss itself is not shown in this range.

**Contraction**
- $\gamma$-contraction $T$ on a complete space, $0<\gamma<1$: unique fixed point (Banach), $\|V_k-V^\star\|\le\gamma^k\|V_0-V^\star\|$; $k\ge\log(C/\epsilon)/(-\log\gamma)$ suffices. Proof: two fixed points are within $\gamma$ times their distance; existence by completeness (p573).

## Notation and objective conventions (p574)
- $\gamma$: discount in $[0,1)$ unless finite horizon is explicit.
- $G_t$: random return from time $t$.
- $V^\pi,Q^\pi$: conditional expected return under $\pi$.
- $\beta$: behavior policy when used as a policy; KL coefficient when scalar.
- $\pi_{old}$: frozen sampling policy within an update.
- $\pi_{ref}$: reference policy for KL regularization.
- $u=-Kx$: LQR feedback convention; LQR minimizes cost.

## Reading lists (p575–p577)
**Primary readings for models and data (p575)**
- Haarnoja et al.: Soft actor critic algorithms and applications
- Hafner et al.: Dreamer world models
- Kumar et al.: Conservative Q learning
- Kostrikov et al.: Implicit Q learning
- Nakamoto et al.: Cal QL
- Song et al.: Hybrid RL
- Janner et al.: Planning with diffusion
- Chi et al.: Diffusion Policy

**Primary readings for LLM post training (p576)**
- Rafailov et al.: DPO
- Gao et al.: REBEL
- Shao et al.: DeepSeekMath
- DeepSeek AI: DeepSeek R1
- Yu et al.: DAPO
- Liu et al.: Dr GRPO analysis
- Gao et al.: REFUEL
- Zhou et al.: Q sharp

**Research readings available by November 10 2026 (p577)**
- Oh et al. (2025): DiscoRL
- Zhang and Zhang (October 2026): GGPO, preprint
- Liu et al. (October 2026): SCALE, preprint
- Kang et al. (September 2026): DeliveryGym, preprint
- Agarwal et al. (2021): Statistical uncertainty in RL evaluation
- Reading standard: identify the setting, assumptions, data access, evaluation budget, and claim supported by each result.
(Also cited inline: Abbasi-Yadkori et al. 2011, p515, p519; Auer, Cesa-Bianchi and Fischer 2002, p528.)

