import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# 1. System parameters (LTI model)
# ---------------------------------
m = 1.0       # mass [kg]
b = 0.5       # viscous friction [N*s/m]

A = np.array([
    [0.0,        1.0],
    [0.0, -b / m]
])

B = np.array([
    [0.0],
    [1.0 / m]
])

C = np.array([[1.0, 0.0]])  # measure position only

# Process noise enters as acceleration: G * w_k
G = np.array([
    [0.0],
    [1.0]
])

# ---------------------------------
# 2. Noise settings
# ---------------------------------
accel_noise_std = 0.5    # std dev of acceleration noise [m/s^2]
meas_noise_std  = 0.2    # std dev of position measurement noise [m]

rng = np.random.default_rng(seed=42)

# ---------------------------------
# 3. Simulation settings
# ---------------------------------
dt = 0.01
T  = 10.0
N  = int(T / dt)
t  = np.linspace(0.0, T, N+1)

# ---------------------------------
# 4. Input u(t) (force)
# ---------------------------------
# Example: a pulse of force between 1s and 3s
u = np.zeros(N+1)
u[(t >= 1.0) & (t <= 3.0)] = 1.0  # 1 N pulse

# ---------------------------------
# 5. Storage for true states and measurements
# ---------------------------------
x_true = np.zeros((N+1, 2))   # [p, v]
y_true = np.zeros(N+1)        # true position
y_meas = np.zeros(N+1)        # noisy measurements

# Initial state: p=0, v=0
x_true[0, :] = np.array([0.0, 0.0])

# ---------------------------------
# 6. Simulation loop with process noise
# ---------------------------------
x = np.zeros((N+1, 2))
y = np.zeros((N+1, 2))

x[0, :] = np.array([0.0, 0.0])
# Process noise (on state)
sigma_w_p = 1e-3   # std dev on position dynamics
sigma_w_v = 1e-2   # std dev on velocity dynamics
#Mass function, this is the mass decreasing over time
def mass_function(t):
    return m - 0.2*np.log(t)
# Measurement noise (on output)
sigma_v_p = 1e-2   # std dev on measured position
sigma_v_v = 1e-2   # std dev on measured velocity
# ---------------------------------
# 6a. Noiseless simulation (same model, w_k = 0, v_k = 0)
# ---------------------------------
x_clean = np.zeros((N+1, 2))
y_clean = np.zeros((N+1, 2))
x_clean[0, :] = np.array([0.0, 0.0])

for k in range(N):
    tk = t[k]
    xk = x_clean[k, :].reshape(2, 1)
    uk = u[k]
    mk = mass_function(tk)

    A = np.array([
        [0.0,        1.0],
        [0.0, -b / mk]
    ])

    B = np.array([
        [0.0],
        [1.0 / mk]
    ])

    C = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    # no process noise
    xdot = A @ xk + B * uk
    x_clean[k+1, :] = (xk + dt * xdot).flatten()

    # no measurement noise
    y_clean[k, :] = (C @ xk).flatten()

# last noiseless output
x_last_clean = x_clean[-1, :].reshape(2, 1)
y_clean[-1, :] = (C @ x_last_clean).flatten()

for k in range(N):
    tk = t[k]
    xk = x[k, :].reshape(2, 1)   # (2,1)
    uk = u[k]
    mk = mass_function(tk)

    A = np.array([
        [0.0,        1.0],
        [0.0, -b / mk]
    ])

    B = np.array([
        [0.0],
        [1.0 / mk]
    ])

    C = np.array([
        [1.0, 0.0],  # output 1 = position
        [0.0, 1.0]   # output 2 = velocity
    ])

    # ----- process noise w_k ~ N(0, Q) -----
    w_k = np.array([
        np.random.normal(0.0, sigma_w_p),
        np.random.normal(0.0, sigma_w_v)
    ]).reshape(2, 1)   # (2,1)

    # state derivative
    xdot = A @ xk + B * uk

    # Euler integration with process noise
    x[k+1, :] = (xk + dt * xdot + w_k).flatten()

    # ----- measurement noise v_k ~ N(0, R) -----
    v_k = np.array([
        np.random.normal(0.0, sigma_v_p),
        np.random.normal(0.0, sigma_v_v)
    ])   # (2,)

    # noiseless output: C x_k   (2x2 @ 2x1 -> 2x1)
    y_noiseless = (C @ xk).flatten()   # (2,)

    # noisy measurement
    y[k, :] = y_noiseless + v_k

# last output (same pattern)
x_last = x[-1, :].reshape(2, 1)
y_last_noiseless = (C @ x_last).flatten()
v_last = np.array([
    np.random.normal(0.0, sigma_v_p),
    np.random.normal(0.0, sigma_v_v)
])
y[-1, :] = y_last_noiseless + v_last

p_ltv = y[:, 0]
v_ltv = y[:, 1]
p_clean = y_clean[:, 0]
v_clean = y_clean[:, 1]

# ---------------------------------
# 8. Plots: true vs noisy "paths"
# ---------------------------------

# ---------------------------------
# 8. Plots: noiseless vs noisy paths
# ---------------------------------

# Position
plt.figure()
plt.plot(t, p_clean, label='Position (noiseless / true)')
plt.plot(t, p_ltv, '--', label='Position (noisy measurement)')
plt.xlabel('Time [s]')
plt.ylabel('Position')
plt.title('Position: noiseless vs noisy')
plt.grid(True)
plt.legend()

# Velocity
plt.figure()
plt.plot(t, v_clean, label='Velocity (noiseless / true)')
plt.plot(t, v_ltv, '--', label='Velocity (noisy measurement)')
plt.xlabel('Time [s]')
plt.ylabel('Velocity')
plt.title('Velocity: noiseless vs noisy')
plt.grid(True)
plt.legend()

plt.show()

