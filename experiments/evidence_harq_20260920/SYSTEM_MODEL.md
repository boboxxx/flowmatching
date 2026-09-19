# Mathematical specification

## 1. Source, packets, observations

Let X in [0,1]^(3xHxW) have source distribution P_X. The terminals share learned
weights, image dimensions, packet partition and quantizer. The sender observes X;
the receiver initially observes only O_0=(Y_0,gamma,Q16(a_0)). The encoder produces
U_0=E_0(X) and G=4 innovation groups U_j=[E_1(X,E_0(X))]_j. U_j is not a repeat
of U_0 and is not conditioned on feedback channel output. A request reveals only
a finite subset index S, not the receiver's latent state.

For n_j complex channel symbols, define

    a_j² = ||U_j||²/n_j,
    S_j = pair(U_j/a_j),
    Y_j = S_j + N_j,       N_j ~ CN(0, gamma^-1 I),
    Z_j = Q16(a_j) unpair(Y_j).

Thus average complex-symbol energy is one for nonzero packets. Q16(a_j), not a_j,
is the receiver's scale. An ideal reliable control link carries quantized scales and
request bits at a declared finite spectral efficiency. The current experiment does
not claim this link is an implemented error-correcting code. Phase/fading, when added,
must replace Y_j with H_j S_j+N_j and condition on estimated rather than free true CSI.

The measured cost is C(S)=n_0+sum_(j in S)n_j+n_scales(S)+n_feedback(S).
Report forward uses separately from reverse uses. CBR=C(S)/(3HW), with no mixing
of real scalar counts and complex symbol counts. Maximum physical rounds: two.

## 2. Observation-measurable actions

A valid acquisition policy pi maps O_0 to S. If packets are requested sequentially,
pi_t must be measurable with respect to sigma(O_0,{Y_j,Q16(a_j): j already received}).
It cannot depend on X, target errors, unreceived enhancement packets, or future noises.
The all-subsets enumerator intentionally violates this constraint and is explicitly
labelled clairvoyant. Its saving is an upper bound only within the fixed finite
action/decoder family and its per-trial quality constraints/failure-fallback rule.
It is not a global optimum under only population-average distortion/outage constraints,
nor an achievable algorithm. In particular, the gate does not optimize deliberate
redistribution of outages or dropping of failed images.

For image estimator a, define distortion d(X,a)=||X-a||²/(3HW) and perceptual
distortion l(X,a)=LPIPS(X,a). The service event is

    F(X,a) = 1{PSNR(X,a)<P0 or l(X,a)>L0}.

The operational objective is min_pi E[C(S_pi)] subject to final outage E[F]<=beta,
and separately a conditional false-ACK constraint P(F=1|ACK)<=alpha at useful ACK
coverage. These constraints are different: sending every packet may still fail service.
MSE, PSNR, MS-SSIM and LPIPS frontiers must accompany threshold-specific outcomes.

## 3. Posterior target and value of evidence

A frozen codec induces the joint distribution P(U_1:G,O_0). Conditional flow matching
learns q_theta(U_1:G|O_0), after training-only standardization. With O_0 fixed:

    epsilon ~ N(0,I), t ~ Uniform(0,1),
    U_t=(1-t)epsilon+t U,
    L_CFM=E||v_theta(U_t,t,O_0)-(U-epsilon)||².

At population optimum and sufficient expressivity, the conditional velocity field
transports the base distribution toward the conditional target distribution; finite
optimization and discretization do not guarantee calibration. Euler sample diversity
is not proof that posterior uncertainty is correct. Repeat independent initial noise
with identical evidence; do not resample the channel while estimating disagreement.

For posterior samples U^(1:M), block uncertainty is

    U_j_hat = 1/(M-1) sum_m ||U_j^(m)-mean_m U_j^(m)||².

Let R(o)=inf_a E[d(X,a)|O=o]. The one-step Bayesian acquisition value is

    V_j(o)=[R(o)-E_(Y_j,Q16(a_j)|o) R(o,Y_j,Q16(a_j))]/DeltaC_j(o).

This expectation is prospective. Once image decoding is nonlinear or risk includes
LPIPS/outage, token variance is not generally proportional to V_j. A variance-ranked
request policy must therefore be compared with deterministic utility prediction and
the actual marginal reductions. Source-aware labels may train such a predictor on
training images, but labels are never inputs at test time.

## 4. Risk certificate and its limits

Freeze score model and L candidate thresholds independently of calibration. For each
threshold tau, let n_tau be the number of accepted independent calibration images
and k_tau their failures. A simultaneous one-sided Clopper-Pearson upper bound is

    B_tau = BetaQuantile(1-delta/L; k_tau+1,n_tau-k_tau),

with B_tau=1 for n_tau=0 or k_tau=n_tau. Choose among thresholds with B_tau<=alpha;
if none qualify, abstain from claiming a certificate. Under iid source/channel pairs,
with probability >=1-delta over calibration, all certified fixed thresholds satisfy
the population conditional false-ACK bound. This is not P(F|O=o)<=alpha for every o.
Each image contributes one preregistered channel draw; repeated draws are correlated
through the source and cannot be counted as independent calibration images.

Even with k=0 and L=1, B=1-delta^(1/n). At n=40, delta=.05, B≈.0722>.05.
The current 40-image split is therefore a calibration pilot, not sufficient evidence
for a 5% certificate. Shifted datasets/channels require a new guarantee or an explicitly
empirical robustness evaluation. Neither posterior covariance nor a low quality-head
score substitutes for this calibration.

## 5. Optional computation action

Only after all previous stages pass, compare ACK, extra inference and transmission
using measured wall-clock latency or operation counts. More solver steps approximate
the same posterior; they do not create new evidence or intrinsically reduce epistemic
uncertainty. A compute action requires demonstrated numerical-error reduction at fixed
observations and random initialization. Optimizing C_air+lambda_t*T_receiver is a
declared extension, not a completed contribution in the present feasibility study.
