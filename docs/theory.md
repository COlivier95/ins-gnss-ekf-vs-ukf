# Theory notes

This page gives the equations implemented in `src/insnav`, in the order the data flows through them.
Every equation below has a direct counterpart in the code (file names in brackets), and the Jacobians are
checked against finite differences in `tests/test_models.py`.

References: P. D. Groves, *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*,
2nd ed. (2013) — J. Solà, *Quaternion kinematics for the error-state Kalman filter* (2017) —
S. Julier & J. Uhlmann, *Unscented filtering and nonlinear estimation*, Proc. IEEE (2004).

---

## 1. Frames and conventions

| Symbol | Meaning |
|---|---|
| $b$ | body frame of the IMU: $x$ forward, $y$ right, $z$ down (FRD) |
| $n$ | local navigation frame: North, East, Down (NED) |
| $e$, $i$ | Earth-fixed (ECEF) and inertial frames |
| $(\varphi, \lambda, h)$ | geodetic latitude, longitude, WGS84 ellipsoidal height |
| $q_{nb}$, $C_{nb}$ | attitude: rotates body vectors into NED, $v^n = C_{nb}\, v^b$ |
| $\otimes$ | Hamilton quaternion product, scalar first $q = [w, x, y, z]$ |
| $[a]_\times$ | skew-symmetric matrix, $[a]_\times b = a \times b$ |
| $q(\phi)$ | quaternion of the rotation vector $\phi$ (exponential map) |

Euler angles are aerospace ZYX: $C_{nb} = R_z(\psi) R_y(\theta) R_x(\phi)$.
Geodesy (`geodesy/wgs84.py`): WGS84 ellipsoid, meridian and transverse radii $R_N$, $R_E$, normal gravity
from the Somigliana formula with a second-order height correction, exact LLA ↔ ECEF ↔ NED conversions.

## 2. Sensor models

$$\tilde f^b = f^b + b_a + n_a, \qquad \tilde\omega^b_{ib} = (I + \mathrm{diag}(s_g))\,\omega^b_{ib} + b_g + n_g$$

$b_a$, $b_g$: accelerometer and gyro biases; $s_g$: gyro scale-factor errors; $n_a$, $n_g$: white noises.
The corrected quantities used everywhere are
$\hat f^b = \tilde f^b - \hat b_a$ and $\hat\omega^b_{ib} = (\tilde\omega^b_{ib} - \hat b_g)(1 - \hat s_g)$ (element-wise).

GNSS (loosely coupled): antenna position, antenna velocity and dual-antenna baseline azimuth, each with the
1σ reported by the receiver.

## 3. Strapdown mechanization (`mechanization/strapdown.py`)

Continuous-time equations in NED (Groves ch. 5):

$$\dot q_{nb} = \tfrac12\, q_{nb} \otimes [0,\ \omega^b_{ib}] - \tfrac12\, [0,\ \omega^n_{in}] \otimes q_{nb}, \qquad
\omega^n_{in} = \omega^n_{ie} + \omega^n_{en}$$

$$\dot v^n = C_{nb} f^b + g^n - (2\,\omega^n_{ie} + \omega^n_{en}) \times v^n$$

$$\dot\varphi = \frac{v_N}{R_N + h}, \qquad \dot\lambda = \frac{v_E}{(R_E + h)\cos\varphi}, \qquad \dot h = -v_D$$

with $\omega^n_{ie} = \Omega\,[\cos\varphi,\ 0,\ -\sin\varphi]^T$ and
$\omega^n_{en} = \left[\frac{v_E}{R_E+h},\ -\frac{v_N}{R_N+h},\ -\frac{v_E \tan\varphi}{R_E+h}\right]^T$.

Discretisation over one IMU interval $\Delta t$ = 5 ms, with the IMU rates averaged over the interval:

$$q_k = q(-\omega^n_{in}\Delta t) \otimes q_{k-1} \otimes q(\hat\omega^b_{ib}\Delta t)$$

$$v_k = v_{k-1} + \left[C_{nb}^{(k-\frac12)} \hat f^b + g^n - (2\omega^n_{ie} + \omega^n_{en}) \times v_{k-1}\right]\Delta t$$

where the mid-interval attitude applies *both* half rotations,
$q^{(k-\frac12)} = q(-\tfrac12\omega^n_{in}\Delta t) \otimes q_{k-1} \otimes q(\tfrac12\hat\omega^b_{ib}\Delta t)$
(omitting the navigation-frame half step biases $f^n$ by $g\,|\omega_{in}|\,\Delta t/2$, i.e. 0.3 m of drift in
10 minutes — caught by the static test). Position is integrated with the trapezoidal velocity. Coning and
sculling corrections are negligible at 200 Hz on a car.

The implementation is written component-wise ("structure of arrays") so that the same function propagates a
single state with fast scalar arithmetic (ES-EKF, 9 µs/step) or all the UKF sigma points in one vectorised call.

## 4. Error state (`models/error_state.py`)

The filters estimate the error of the nominal (total) state, which stays small and lives in a vector space:

$$\delta x = [\ \delta r^n\ (3),\ \delta v^n\ (3),\ \delta\theta\ (3),\ \delta b_a\ (3),\ \delta b_g\ (3),\ \delta s_g\ (3),\ \delta p_g\ (3),\ \delta\psi_b\ (1)\ ] \in \mathbb{R}^{22}$$

| Block | Definition |
|---|---|
| $\delta r$ | position error in NED [m] |
| $\delta v$ | velocity error in NED |
| $\delta\theta$ | attitude error in NED: $C_{nb} = (I + [\delta\theta]_\times)\,\hat C_{nb}$, $q_{nb} = q(\delta\theta) \otimes \hat q_{nb}$ |
| $\delta b_a$, $\delta b_g$, $\delta s_g$ | sensor biases and gyro scale factors (random walks) |
| $\delta p_g$ | time-correlated GNSS position error (first-order Gauss-Markov), colored measurement noise |
| $\delta\psi_b$ | dual-antenna heading error: baseline misalignment + slow multipath wander (random walk) |

**Injection** $x = \hat x \oplus \delta x$: positions through the radii of curvature, attitude by
left-multiplication with $q(\delta\theta)$, everything else additively. **Difference** $\ominus$ is its inverse
(used by the UKF to average sigma points).

### Linearised dynamics

$$\dot{\delta x} = F\,\delta x + G\,w$$

$$\begin{aligned}
\dot{\delta r} &= \delta v \\
\dot{\delta v} &= -[f^n]_\times \delta\theta - C_{nb}\,\delta b_a - (2\omega_{ie} + \omega_{en}) \times \delta v + \tfrac{2g}{R}\,\delta r_D\, e_D - C_{nb}\, n_a \\
\dot{\delta\theta} &= -\omega_{in} \times \delta\theta - C_{nb}(1 - s_g)\,\delta b_g - C_{nb}\,\mathrm{diag}(\tilde\omega - \hat b_g)\,\delta s_g - C_{nb}\, n_g \\
\dot{\delta b_a} &= w_{ba}, \quad \dot{\delta b_g} = w_{bg}, \quad \dot{\delta s_g} = w_{sg}, \quad \dot{\delta\psi_b} = w_{\psi} \\
\dot{\delta p_g} &= -\delta p_g / \tau_g + w_{pg}
\end{aligned}$$

with $f^n = \hat C_{nb}\hat f^b$. The $2g/R$ term is the (unstable) vertical-channel gravity feedback.

**Consistency between nominal and error models.** Random-walk states keep a constant nominal value between
updates. The Gauss-Markov state must decay in the nominal state too, $\dot{\hat p}_g = -\hat p_g/\tau_g$
(`propagate_nominal_aux`); forgetting it makes the error model and the nominal state disagree and the
estimate drifts (it did: 1.9 m in 20 minutes).

### Discretisation

$\Phi = I + F\Delta t$ (200 Hz) and, because the IMU noise is isotropic, $C_{nb}$ drops out of
$G Q_c G^T$ and the process noise is diagonal:

$$Q_d = \mathrm{diag}\!\left(0_3,\ \sigma_{VRW}^2,\ \sigma_{ARW}^2,\ \tfrac{2\sigma_{ba}^2}{\tau_{ba}},\ \tfrac{2\sigma_{bg}^2}{\tau_{bg}},\ q_{sg},\ \tfrac{2\sigma_{pg}^2}{\tau_{pg}},\ q_{\psi}\right)\Delta t$$

## 5. Measurement models (`models/measurements.py`)

Each model provides a nonlinear prediction $h(x)$ (used by both filters), a residual in a vector space and
its Jacobian $H = \partial\,\mathrm{residual} / \partial\,\delta x$ (ES-EKF only). $l$ is the IMU → primary
antenna lever arm, $b^b$ the primary → secondary baseline, both in the body frame.

**Position** — the GNSS position is the antenna position plus the correlated error:

$$z_p = (\varphi, \lambda, h)_{IMU} \oplus \left(C_{nb}\, l + p_g\right) + w_p, \qquad
H_p = [\ I_3,\ 0,\ -[C_{nb} l]_\times,\ 0,\ 0,\ 0,\ I_3,\ 0\ ]$$

The residual $z - h(x)$ is expressed in NED metres.

**Velocity** — lever-arm effect of the rotation (Earth-rate term < 0.1 mm/s, neglected):

$$z_v = v^n + C_{nb}(\hat\omega^b_{ib} \times l) + w_v$$

$$H_v = \left[\ 0,\ I_3,\ -[C_{nb}(\hat\omega \times l)]_\times,\ 0,\ C_{nb}[l]_\times (1 - s_g),\ C_{nb}[l]_\times \mathrm{diag}(\tilde\omega - \hat b_g),\ 0,\ 0\ \right]$$

**Dual-antenna heading** — azimuth of the baseline in NED, which handles the +90° mounting (baseline along
$y$) and the roll / pitch coupling of an inclined baseline without special cases:

$$b^n = C_{nb}\, b^b, \qquad z_\psi = \operatorname{atan2}(b^n_E,\ b^n_N) + \psi_b + w_\psi$$

$$H_\psi = \left[\ 0,\ 0,\ \tfrac{\partial \psi}{\partial b^n}\,(-[b^n]_\times),\ 0,\ 0,\ 0,\ 0,\ 1\ \right], \qquad
\tfrac{\partial \psi}{\partial b^n} = \frac{[-b^n_E,\ b^n_N,\ 0]}{(b^n_N)^2 + (b^n_E)^2}$$

The residual is wrapped to $(-\pi, \pi]$.

## 6. ES-EKF (`filters/es_ekf.py`)

**Prediction** (every IMU sample): nominal state through the nonlinear mechanization,
$P \leftarrow \Phi P \Phi^T + Q_d$.

**Update** (every GNSS measurement):

$$y = z \ominus h(\hat x), \quad S = H P H^T + R, \quad K = P H^T S^{-1}, \quad \delta x = K y$$

$$P \leftarrow (I - KH)\,P\,(I - KH)^T + K R K^T \quad \text{(Joseph form)}$$

then **injection** $\hat x \leftarrow \hat x \oplus \delta x$ and **reset** of the error state to zero with
$P \leftarrow G P G^T$, $G = \mathrm{blkdiag}(I, I, I - \tfrac12[\delta\theta]_\times, I, \dots)$.

## 7. ES-UKF (`filters/es_ukf.py`)

Same nominal state, error state, noise and measurement models. The sigma points live in the **error space**
and are mapped onto the manifold by the injection, so no quaternion is ever averaged.

**Weights** (scaled unscented transform, $n = 22$):
$\lambda = \alpha^2(n + \kappa) - n$, $\gamma = \sqrt{n + \lambda}$,
$W_0^m = \lambda/(n+\lambda)$, $W_0^c = W_0^m + 1 - \alpha^2 + \beta$, $W_i^m = W_i^c = 1/(2(n+\lambda))$.
With $\alpha = 1, \beta = 2, \kappa = 0$ all weights are non-negative (the covariance stays positive
semi-definite by construction) and the points sit at $\sqrt{n} \approx 4.7\sigma$.

**Sigma points**: $\chi_0 = 0$, $\chi_{\pm j} = \pm\gamma\, S_{:,j}$ with $P = S S^T$ (Cholesky), and
$\mathcal{X}_i = \hat x \oplus \chi_i$.

**Time update.** Every sigma point is propagated through the full nonlinear mechanization with its *own*
biases and scale factors (all 45 points in one vectorised call). The points are regenerated every 20 IMU
samples (10 Hz) and before every GNSS update; the nominal state is still propagated at 200 Hz for the outputs.

$$\delta_i = \mathcal{X}_i^- \ominus \mathcal{X}_0^-, \quad \bar\delta = \sum W_i^m \delta_i, \quad
P^- = \sum W_i^c (\delta_i - \bar\delta)(\delta_i - \bar\delta)^T + \textstyle\sum Q_d, \quad
\hat x^- = \mathcal{X}_0^- \oplus \bar\delta$$

**Measurement update** (no Jacobian):

$$\mathcal{Z}_i = h(\mathcal{X}_i), \quad \bar z = \mathcal{Z}_0 + \sum W_i^m (\mathcal{Z}_i \ominus \mathcal{Z}_0)$$

$$P_{zz} = \sum W_i^c\, e_i e_i^T + R, \quad P_{xz} = \sum W_i^c\, \chi_i e_i^T, \quad K = P_{xz} P_{zz}^{-1}$$

$$\delta x = K (z \ominus \bar z), \quad P \leftarrow P - K P_{zz} K^T$$

followed by the same injection and reset as the ES-EKF.

## 8. Shared machinery (`filters/base.py`)

**Exact time tagging.** GNSS epochs are processed at their measurement time (`gpsTow`). When an epoch falls
inside an IMU interval, the interval is split, the update applied, and the propagation resumed — a 5 ms
mismatch at 20 m/s would otherwise be a 10 cm error.

**Coarse alignment** (`alignment/coarse.py`), with no information from the reference: roll and pitch from
the mean specific force during the first 5 s at rest, heading by inverting the baseline geometry on the
first dual-antenna measurement, position and velocity from the first GNSS fix (moved to the IMU).

**Innovation gating.** Each measurement is tested with the normalised innovation squared
$\mathrm{NIS} = y^T S^{-1} y$ against $\chi^2_{m}(0.999)$ ($m$ = measurement dimension) and rejected above it.

**Re-acquisition.** After 10 consecutive rejections of the GNSS *position* (2 s), the filter inflates its
position covariance by the squared innovation and accepts the fix. Without it, a filter that drifted more
than it believes during a long outage rejects the GNSS forever. Velocity and heading are never forced.

## 9. Evaluation (`evaluation/`)

Errors are computed against the reference interpolated at the filter epochs (yaw unwrapped before
interpolation). The reference itself has an uncertainty (Qinertia exports its 1σ), so the consistency
of the filter is tested on the difference filter − reference with the combined covariance:

$$\sigma_{comb} = \sqrt{\sigma_{filter}^2 + \sigma_{ref}^2}, \qquad
\text{inside } 3\sigma = \Pr\left(|e| \le 3\sigma_{comb}\right), \qquad
\text{ratio} = \sqrt{\overline{(e / \sigma_{comb})^2}}$$

A consistent filter has ~99.7 % of its errors inside $3\sigma$ (Gaussian) and a ratio close to 1; a ratio well
above 1 means overconfidence, well below 1 means a conservative filter.
