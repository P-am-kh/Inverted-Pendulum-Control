import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation


# 1. Define the Dynamics for an INVERTED Pendulum
def inverted_pendulum_dynamics(t, x, u_val):
    """
    Defines the dynamics of an inverted pendulum.

    Args:
        t (float): Time.
        x (list): State vector [theta, theta_dot].
        u_val (float): External control input (torque).

    Returns:
        list: Derivative of the state vector [theta_dot, theta_ddot].
    """
    theta, theta_dot = x

    global g, L, b

    # Key change: The sign of the gravitational term is flipped for an inverted pendulum
    # If theta=0 is downward, and theta=pi is upward, then sin(theta) is positive
    # near 0, and negative near pi. Gravity should push it away from pi.
    # So, if theta is slightly > pi, sin(theta) is negative, we want theta_ddot to be positive.
    # Thus, -(g / L) * np.sin(theta)
    theta_ddot = -(g / L) * np.sin(theta) - b * theta_dot + u_val

    return [theta_dot, theta_ddot]


# 2. Set Parameters
g = 9.81  # m/s^2 (acceleration due to gravity)
L = 1.0  # m (length of the pendulum rod)
b = 0.1  # Nms/rad (damping coefficient)

# Initial conditions for an INVERTED pendulum
# Start slightly off the upright (unstable) equilibrium (pi radians)
theta0 = np.pi - 0.1  # Slightly less than 180 degrees (upright)
theta_dot0 = 0.0  # Initial angular velocity
x0 = [theta0, theta_dot0]

# Time span
t_span = (0, 10)  # Simulate from t=0 to t=10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Points at which to store the solution

# Control input (u) - No external torque for now
u_constant = 0.0


# Create a wrapper function for solve_ivp
def ode_system_inverted(t, x):
    return inverted_pendulum_dynamics(t, x, u_constant)


# 3. Solve the ODE
sol = solve_ivp(ode_system_inverted, t_span, x0, t_eval=t_eval, rtol=1e-5, atol=1e-7)

# Extract results
time = sol.t
theta = sol.y[0]
theta_dot = sol.y[1]

# 4. Calculate Pendulum Position (x, y coordinates)
# The pivot point is at (0, 0)
x_coords = L * np.sin(theta)
y_coords = -L * np.cos(theta)

# 5. Visualize (Animation)
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-(L + 0.2), L + 0.2)
ax.set_ylim(-(L + 0.2), L + 0.2)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Inverted Pendulum Simulation")
ax.grid(True)

# Plot the pivot point
ax.plot(0, 0, "o", color="red", markersize=8, label="Pivot")

# Line representing the pendulum rod
(line,) = ax.plot([], [], "o-", lw=2, markersize=10, label="Pendulum")

# Text for time display
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text("")
    return line, time_text


def animate(i):
    # Update the pendulum rod and bob position
    x_val = [0, x_coords[i]]
    y_val = [0, y_coords[i]]
    line.set_data(x_val, y_val)

    # Update time text
    time_text.set_text(f"Time: {time[i]:.2f} s")

    return line, time_text


# Create the animation
ani = FuncAnimation(
    fig, animate, frames=len(time), init_func=init, blit=True, interval=sol.t[1] * 1000
)

plt.legend()
plt.show()

# Plot the state variables over time:
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, np.degrees(theta))
plt.title("Inverted Pendulum Angle over Time")
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, np.degrees(theta_dot))
plt.title("Inverted Pendulum Angular Velocity over Time")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (degrees/s)")
plt.grid(True)

plt.tight_layout()
plt.show()
