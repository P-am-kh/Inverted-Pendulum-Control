import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

# System parameters
m = 0.1  # pendulum mass (kg)
M = 1.0  # cart mass (kg)
L = 0.5  # pendulum length (m)
g = 9.81  # gravity (m/s²)
b = 0.1  # friction coefficient

# Controller gains
Kp = 200.0
Kd = 40.0

# Initial conditions
theta0 = np.pi + 0.5  # initial angle (0.5 rad ≈ 28.6° from vertical)
theta_dot0 = 0.0
x0 = 0.0
x_dot0 = 0.0

# Simulation parameters
t_start = 0.0
t_end = 10.0
dt = 0.05
t_points = np.arange(t_start, t_end, dt)


def inverted_pendulum(t, y):
    """System dynamics with PD control"""
    theta, theta_dot, x, x_dot = y

    # Normalize angle
    error = (theta - np.pi + np.pi) % (2 * np.pi) - np.pi

    # Control force with saturation
    F = -Kp * error - Kd * theta_dot
    F = np.clip(F, -30, 30)

    # Equations of motion
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denominator = M + m - m * cos_theta**2

    theta_ddot = (
        m * g * sin_theta
        - m * cos_theta * (F + m * L * theta_dot**2 * sin_theta)
        - b * theta_dot
    ) / (L * denominator)
    x_ddot = (
        F + m * L * (theta_dot**2 * sin_theta - theta_ddot * cos_theta) - b * x_dot
    ) / denominator

    return [theta_dot, theta_ddot, x_dot, x_ddot]


# Solve ODE
sol = solve_ivp(
    inverted_pendulum,
    [t_start, t_end],
    [theta0, theta_dot0, x0, x_dot0],
    t_eval=t_points,
    method="RK45",
    rtol=1e-8,
    atol=1e-10,
)

# Process results
theta = np.unwrap(sol.y[0])
theta_dot = sol.y[1]
x = savgol_filter(sol.y[2], 21, 3)  # smooth position
x_dot = savgol_filter(sol.y[3], 21, 3)  # smooth velocity

# Create figure with two subplots
plt.figure(figsize=(14, 8))
gs = plt.GridSpec(2, 2, height_ratios=[2, 1])

# Animation subplot
ax1 = plt.subplot(gs[0, :])
ax1.set_xlim(-4, 2)
ax1.set_ylim(-1, 1)
ax1.set_aspect("equal")
ax1.grid(True)

# Create animated elements
cart_width, cart_height = 0.4, 0.2
cart = Rectangle(
    (x[0] - cart_width / 2, -cart_height / 2),
    cart_width,
    cart_height,
    fc="blue",
    ec="black",
)
ax1.add_patch(cart)

(pendulum,) = ax1.plot(
    [x[0], x[0] + L * np.sin(theta[0])], [0, -L * np.cos(theta[0])], "r-", lw=3
)
(bob,) = ax1.plot(x[0] + L * np.sin(theta[0]), -L * np.cos(theta[0]), "ro", ms=10)
time_text = ax1.text(
    0.02, 0.95, "", transform=ax1.transAxes, bbox=dict(facecolor="white", alpha=0.8)
)

# States plot
ax2 = plt.subplot(gs[1, 0])
ax3 = plt.subplot(gs[1, 1])

# Plot theta and theta_dot
ax2.plot(t_points, np.degrees(theta - np.pi), "b-", label="Angle [deg]")
ax2.plot(t_points, theta_dot, "r-", label="Angular velocity [rad/s]")
ax2.set_xlabel("Time [s]")
ax2.set_title("Pendulum States")
ax2.legend()
ax2.grid(True)

# Plot x and x_dot
ax3.plot(t_points, x, "g-", label="Cart position [m]")
ax3.plot(t_points, x_dot, "m-", label="Cart velocity [m/s]")
ax3.set_xlabel("Time [s]")
ax3.set_title("Cart States")
ax3.legend()
ax3.grid(True)

plt.tight_layout()


# Animation update function
def update(frame):
    # Update animation
    cart.set_xy((x[frame] - cart_width / 2, -cart_height / 2))
    pendulum.set_data(
        [x[frame], x[frame] + L * np.sin(theta[frame])], [0, -L * np.cos(theta[frame])]
    )
    bob.set_data([x[frame] + L * np.sin(theta[frame])], [-L * np.cos(theta[frame])])

    # Update time info
    angle_deg = np.degrees((theta[frame] - np.pi + np.pi) % (2 * np.pi) - np.pi)
    time_text.set_text(
        f"Time: {t_points[frame]:.2f}s\n"
        f"Angle: {angle_deg:.1f}°\n"
        f"Cart Pos: {x[frame]:.2f}m"
    )

    # Update view limits smoothly
    margin = 2.0
    current_x = x[frame]
    current_xlim = ax1.get_xlim()

    if (
        current_x < current_xlim[0] + 0.3 * margin
        or current_x > current_xlim[1] - 0.3 * margin
    ):
        ax1.set_xlim(current_x - margin, current_x + margin)

    # Highlight current time in state plots
    for ax in [ax2, ax3]:
        for artist in ax.lines + ax.collections:
            if hasattr(artist, "_current_time"):
                artist.remove()

    # Vertical line for current time
    ax2.axvline(
        t_points[frame], color="k", linestyle="--", alpha=0.5, label="_current_time"
    )
    ax3.axvline(
        t_points[frame], color="k", linestyle="--", alpha=0.5, label="_current_time"
    )

    return cart, pendulum, bob, time_text


# Create animation
ani = FuncAnimation(
    plt.gcf(),
    update,
    frames=len(t_points),
    init_func=lambda: [cart, pendulum, bob, time_text],
    interval=20,
    blit=False,
)

plt.suptitle("Inverted Pendulum: Animation and State Evolution", y=1.02)
plt.show()
