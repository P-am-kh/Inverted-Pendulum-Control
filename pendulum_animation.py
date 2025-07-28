import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation


# 1. Define the Dynamics (as you provided)
def pendulum_dynamics(t, x, u_val):
    """
    Defines the dynamics of a simple pendulum.

    Args:
        t (float): Time.
        x (list): State vector [theta, theta_dot].
        u_val (float): External control input (torque).

    Returns:
        list: Derivative of the state vector [theta_dot, theta_ddot].
    """
    theta, theta_dot = x

    # Ensure u_val is used, even if it's a function of t
    # For simplicity, we'll assume u_val is a constant for now or handled externally

    # Pendulum parameters (global for this example, or pass as args if needed)
    global g, L, b

    theta_ddot = (
        -(g / L) * np.sin(theta) - b * theta_dot + u_val
    )  # Corrected sign for u
    # Note: I've changed the sign of u based on standard convention where positive u
    # would typically try to increase theta (clockwise torque if theta is clockwise).
    # If your u is a braking force, then your original sign might be correct.
    # Just be mindful of how you define your control input.

    return [theta_dot, theta_ddot]


# 2. Set Parameters
g = 9.81  # m/s^2 (acceleration due to gravity)
L = 1.0  # m (length of the pendulum rod)
b = 0.1  # Nms/rad (damping coefficient)

# Initial conditions
theta0 = np.pi / 2  # Initial angle (90 degrees)
theta_dot0 = 0.0  # Initial angular velocity
x0 = [theta0, theta_dot0]

# Time span
t_span = (0, 50)  # Simulate from t=0 to t=10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Points at which to store the solution

# Control input (u) - For this example, let's keep it simple
# You could define a function for u if it varies with time: u_func(t)
u_constant = 0.0  # No external torque for now


# Create a wrapper function for solve_ivp to handle the constant u
def ode_system(t, x):
    return pendulum_dynamics(t, x, u_constant)


# 3. Solve the ODE
sol = solve_ivp(ode_system, t_span, x0, t_eval=t_eval, rtol=1e-5, atol=1e-7)

# Extract results
time = sol.t
theta = sol.y[0]
theta_dot = sol.y[1]

# 4. Calculate Pendulum Position (x, y coordinates)
# The pivot point is at (0, 0)
x_coords = L * np.sin(theta)
y_coords = -L * np.cos(theta)  # Negative because y-axis typically points up,
# and a pendulum hanging down has negative y

# 5. Visualize (Animation)
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-(L + 0.2), L + 0.2)
ax.set_ylim(-(L + 0.2), L + 0.2)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Pendulum Simulation")
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
)  # interval in ms

plt.legend()
plt.show()

# You can also plot the state variables over time:
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, np.degrees(theta))
plt.title("Pendulum Angle over Time")
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time, np.degrees(theta_dot))
plt.title("Pendulum Angular Velocity over Time")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (degrees/s)")
plt.grid(True)

plt.tight_layout()
plt.show()
