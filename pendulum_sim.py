import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from control import lqr
import os


os.makedirs("plots", exist_ok=True)


# Parameters
g = 9.81  # Gravity (m/s²)
L = 1.0  # Pendulum length (m)
m = 1.0  # Mass (kg)
b = 0.1  # Damping coefficient

# Simulation time
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

# Initial condition: [theta, theta_dot]
x0 = [np.pi - 0.1, 0]  # Slightly offset from upright


# Dynamics
def pendulum_dynamics(t, x, u):
    theta, theta_dot = x
    theta_ddot = -(g / L) * np.sin(theta) - b * theta_dot - u
    return [theta_dot, theta_ddot]


# PID Controller
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0
        self.last_t = None  # New: store last time

    def compute(self, error, current_t):

        if self.last_t is None:
            dt = 0.01  # Or some small default
        else:
            dt = current_t - self.last_t

        if dt == 0:  # Avoid division by zero if solve_ivp returns same time
            dt = 1e-9  # A very small non-zero value

        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        self.last_t = current_t  # Update last_t

        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative


# LQR Controller
def design_lqr():
    # Linearized state-space around theta=pi (upright)
    A = np.array([[0, 1], [g / L, -b]])
    B = np.array([[0], [1]])
    Q = np.diag([100, 1])  # State weights (theta, theta_dot)
    R = np.eye(1) * 0.1  # Control effort weight
    K, _, _ = lqr(A, B, Q, R)
    return K


# Simulate with PID
def simulate_pid():
    pid = PIDController(Kp=10, Ki=0, Kd=1)
    # dt = t_eval[1] - t_eval[0]
    u_history = []
    x_history = []
    t_history = []

    def pid_controlled_dynamics(t, x):
        theta = x[0] - np.pi  # Error from upright
        u = pid.compute(theta, t)
        u = np.clip(u, -10, 10)
        u_history.append(u)
        x_history.append(x)
        t_history.append(t)
        return pendulum_dynamics(t, x, u)

    sol = solve_ivp(pid_controlled_dynamics, t_span, x0, t_eval=t_eval)
    return np.array(t_history), np.array(u_history), np.array(x_history)


# Simulate with LQR
def simulate_lqr():
    K = design_lqr()
    u_history = []
    x_history = []
    t_history = []

    def lqr_controlled_dynamics(t, x):
        theta = x[0] - np.pi  # Linearize around upright
        u = -K @ np.array([theta, x[1]])
        u = np.clip(u, -10, 10)
        u_history.append(u[0])
        x_history.append(x)
        t_history.append(t)
        return pendulum_dynamics(t, x, u[0])

    sol = solve_ivp(lqr_controlled_dynamics, t_span, x0, t_eval=t_eval)
    return np.array(t_history), np.array(u_history), np.array(x_history)


# Run simulations
pid_t, pid_u, pid_x = simulate_pid()
lqr_t, lqr_u, lqr_x = simulate_lqr()

# Plot results
plt.figure(figsize=(12, 8))

# Plot angle (theta)
plt.subplot(2, 2, 1)
plt.plot(pid_t, pid_x[:, 0], label="PID")
plt.plot(lqr_t, lqr_x[:, 0], label="LQR")
plt.axhline(np.pi, color="r", linestyle="--", label="Desired")
plt.ylabel("Angle (rad)")
plt.title("Pendulum Angle")
plt.legend()

# Plot control effort (u)
plt.subplot(2, 2, 2)
plt.plot(pid_t, pid_u, label="PID")
plt.plot(lqr_t, lqr_u, label="LQR")
plt.ylabel("Control Input (N·m)")
plt.title("Control Effort")

# Phase portrait (PID)
plt.subplot(2, 2, 3)
plt.plot(pid_x[:, 0], pid_x[:, 1])
plt.xlabel("Angle (rad)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("PID Phase Portrait")

# Phase portrait (LQR)
plt.subplot(2, 2, 4)
plt.plot(lqr_x[:, 0], lqr_x[:, 1])
plt.xlabel("Angle (rad)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("LQR Phase Portrait")

plt.tight_layout()
plt.savefig("plots/pid_vs_lqr.png")
plt.show()
