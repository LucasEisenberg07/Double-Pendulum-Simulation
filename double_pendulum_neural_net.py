import torch.nn as nn
import torch.optim as optim
import torch
import math
import tkinter as tk
import numpy as np
import torch
import os
mouse_has_pendulum_one = False
mouse_has_pendulum_two = False
playing = True
x_origin_offset = 0
y_origin_offset = 0
gravity = -9.81
steps = 500
curr_step = 0
itterations = 200
n_starts = 50
initial_origin = (400 + x_origin_offset, 300 + y_origin_offset)

drag_coefficient1 = 0.0
drag_coefficient2 = 0.0
driving_amplitude1 = 0.0
driving_amplitude2 = 0.0
driving_frequency1 = 0.0
driving_frequency2 = 0.0

final_x2 = 0.0
final_y2 = 0.0

best_loss = 100.0
trained_theta1 = 0.0
trained_theta2 = 0.0
trained_damp1 = 0.0
trained_damp2 = 0.0
trained_amp1 = 0.0
trained_amp2 = 0.0
trained_freq1 = 0.0
trained_freq2 = 0.0

target_x_global = 0.0
target_y_global = 0.0

FAST_OMEGA_THRESHOLD = 5000.0
too_fast = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def integrate_double_pendulum(
    theta1, theta2, omega1, omega2,
    l1, l2, m1, m2, g,
    drag1, drag2, amp1, amp2, freq1, freq2,
    steps, dt, device=None, save_final=False
):
    # Accepts all values as floats or tensors, returns final (theta1, theta2, omega1, omega2, x2, y2)
    # device: if not None, use torch tensors on this device, else use floats
    if device is not None:
        def to_tensor(x, requires_grad=False):
            # Use as_tensor to preserve computation graph if x is already a tensor
            return torch.as_tensor(x, dtype=torch.float32, device=device)
        # Only state variables need gradients in training
        theta1 = to_tensor(theta1).requires_grad_()
        theta2 = to_tensor(theta2).requires_grad_()
        omega1 = to_tensor(omega1).requires_grad_()
        omega2 = to_tensor(omega2).requires_grad_()
        l1 = to_tensor(l1)
        l2 = to_tensor(l2)
        m1 = to_tensor(m1)
        m2 = to_tensor(m2)
        g = to_tensor(g)
        drag1 = to_tensor(drag1)
        drag2 = to_tensor(drag2)
        amp1 = to_tensor(amp1)
        amp2 = to_tensor(amp2)
        freq1 = to_tensor(freq1)
        freq2 = to_tensor(freq2)
        t = to_tensor(0.0)
    else:
        def to_tensor(x, requires_grad=False):
            return torch.as_tensor(x, dtype=torch.float64)
        theta1 = to_tensor(theta1)
        theta2 = to_tensor(theta2)
        omega1 = to_tensor(omega1)
        omega2 = to_tensor(omega2)
        l1 = to_tensor(l1)
        l2 = to_tensor(l2)
        m1 = to_tensor(m1)
        m2 = to_tensor(m2)
        g = to_tensor(g)
        drag1 = to_tensor(drag1)
        drag2 = to_tensor(drag2)
        amp1 = to_tensor(amp1)
        amp2 = to_tensor(amp2)
        freq1 = to_tensor(freq1)
        freq2 = to_tensor(freq2)
        t = to_tensor(0.0)
    for _ in range(steps):
        t = t + dt
        delta = theta2 - theta1
        denom1 = (m1 + m2) * l1 - m2 * l1 * torch.cos(delta) ** 2
        denom2 = (l2 / l1) * denom1
        drive1 = amp1 * torch.sin(2 * math.pi * freq1 * t)
        drive2 = amp2 * torch.sin(2 * math.pi * freq2 * t)
        domega1_dt = (
            (m2 * l1 * omega1 ** 2 * torch.sin(delta) * torch.cos(delta)
             + m2 * g * torch.sin(theta2) * torch.cos(delta)
             + m2 * l2 * omega2 ** 2 * torch.sin(delta)
             - (m1 + m2) * g * torch.sin(theta1)) / denom1
            - drag1 * omega1
            + drive1 / (m1 * l1 ** 2)
        )
        if m2.item() == 0:
            domega2_dt = to_tensor(0.0)
        else:
            domega2_dt = (
                (-m2 * l2 * omega2 ** 2 * torch.sin(delta) * torch.cos(delta)
                 + (m1 + m2) * g * torch.sin(theta1) * torch.cos(delta)
                 - (m1 + m2) * l1 * omega1 ** 2 * torch.sin(delta)
                 - (m1 + m2) * g * torch.sin(theta2)) / denom2
                - drag2 * omega2
                + drive2 / (m2 * l2 ** 2)
            )
        omega1 = omega1 + domega1_dt * dt
        omega2 = omega2 + domega2_dt * dt
        theta1 = theta1 + omega1 * dt
        theta2 = theta2 + omega2 * dt
    x1 = l1 * torch.sin(theta1)
    y1 = -l1 * torch.cos(theta1)
    x2 = x1 + l2 * torch.sin(theta2)
    y2 = y1 - l2 * torch.cos(theta2)
    return theta1, theta2, omega1, omega2, x2, y2

def train_controller():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Choose a random reachable point for the second mass
    l1 = 1.0
    l2 = 1.0
    # The reachable area is a ring: r in [|l1-l2|, l1+l2], theta in [pi, 2pi]
    r = np.random.uniform(1, l1 + l2)
    theta = np.random.uniform(np.pi, 2 * np.pi)
    global target_x_global, target_y_global
    target_x_global = r * np.cos(theta)
    target_y_global = -r * np.sin(theta)  # negative y is up in this convention
    print(f"Target for mass 2: x={target_x_global:.3f}, y={target_y_global:.3f}")

    def run_to_target(params, steps=steps, dt=0.01):
        device = params[0].device if isinstance(params[0], torch.Tensor) else torch.device('cpu')
        theta1 = params[6] * math.pi * 2 - math.pi
        theta2 = params[7] * math.pi * 2 - math.pi
        omega1 = 0.0
        omega2 = 0.0
        l1_val = 1.0
        l2_val = 1.0
        m1_val = 1.0
        m2_val = 1.0
        g_val = gravity
        amp1, amp2, freq1, freq2, drag1, drag2 = params[0], params[1], params[2], params[3], params[4], params[5]
        _, _, omega1f, omega2f, x2, y2 = integrate_double_pendulum(
            theta1, theta2, omega1, omega2,
            l1_val, l2_val, m1_val, m2_val, g_val,
            drag1, drag2, amp1, amp2, freq1, freq2,
            steps, dt, device=device
        )
        
        global trained_theta1, trained_theta2, trained_damp1, trained_damp2, trained_amp1, trained_amp2, trained_freq1, trained_freq2
        trained_theta1 = float(theta1)
        trained_theta2 = float(theta2)
        trained_damp1 = float(drag1)
        trained_damp2 = float(drag2)
        trained_amp1 = float(amp1)
        trained_amp2 = float(amp2)
        trained_freq1 = float(freq1)
        trained_freq2 = float(freq2)

        # Save params as before
        np.save("optimized_params.npy", np.array([
            amp1.item() if hasattr(amp1, 'item') else float(amp1),
            amp2.item() if hasattr(amp2, 'item') else float(amp2),
            freq1.item() if hasattr(freq1, 'item') else float(freq1),
            freq2.item() if hasattr(freq2, 'item') else float(freq2),
            drag1.item() if hasattr(drag1, 'item') else float(drag1),
            drag2.item() if hasattr(drag2, 'item') else float(drag2),
            params[6].item() if hasattr(params[6], 'item') else float(params[6]),
            params[7].item() if hasattr(params[7], 'item') else float(params[7]),
            0,
            0
        ]))
        global final_x2, final_y2
        final_x2, final_y2 = x2, y2
        return x2, y2, omega1f, omega2f
    
    params = torch.nn.Parameter(torch.empty(10, device=device).uniform_(0, 0.5))
    curr_best_params = params.clone()
    curr_best_loss = 100
    optimizer = optim.Adam([params], lr=0.01)
    
    for start in range(n_starts):
        params = torch.nn.Parameter(torch.empty(10, device=device).uniform_(0, 1))
        print(f"Starting nstart run {start+1}/{n_starts}")
        amp1 = params[0].clamp(0, 0)
        amp2 = params[1].clamp(0, 0)
        freq1 = params[2].clamp(0, 2)
        freq2 = params[3].clamp(0, 2)
        drag1 = params[4].clamp(-2, 2)
        drag2 = params[5].clamp(-2, 2)
        theta1_init = params[6].clamp(0, 1)
        theta2_init = params[7].clamp(0, 1)
        param_list = [amp1, amp2, freq1, freq2, drag1, drag2, theta1_init, theta2_init]
        x2, y2, omega1, omega2 = run_to_target(param_list, steps=steps, dt=0.01)
        final_x2, final_y2 = x2, y2
        dist = torch.sqrt((x2 - target_x_global) ** 2 + (y2 - target_y_global) ** 2)
        loss = dist
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    params = curr_best_params.clone().detach().requires_grad_()
    optimizer = optim.Adam([params], lr=0.01)
    for step in range(itterations):
        # Clamp parameters to valid ranges
        amp1 = params[0].clamp(0, 0)
        amp2 = params[1].clamp(0, 0)
        freq1 = params[2].clamp(0, 2)
        freq2 = params[3].clamp(0, 2)
        drag1 = params[4].clamp(0, 2)
        drag2 = params[5].clamp(0, 2)
        theta1_init = params[6].clamp(0, 1)
        theta2_init = params[7].clamp(0, 1)
        param_list = [amp1, amp2, freq1, freq2, drag1, drag2, theta1_init, theta2_init]
        x2, y2, omega1, omega2 = run_to_target(param_list, steps=steps, dt=0.01)
        final_x2, final_y2 = x2, y2
        dist = torch.sqrt((x2 - target_x_global) ** 2 + (y2 - target_y_global) ** 2)
        loss = dist
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Step {step}/{itterations}: Loss = {loss.item():.4f}, target=({target_x_global:.2f},{target_y_global:.2f}), final=({x2.item():.2f},{y2.item():.2f}), final_speeds=({omega1.item():.2f},{omega2.item():.2f})")

class DoublePendulum:
    def __init__(self, m1, m2, l1, l2, theta1, theta2, g=gravity):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2 
        self.g = g

        self.theta1 = theta1
        self.theta2 = theta2
        self.omega1 = 0
        self.omega2 = 0


    def step(self, dt):
        global too_fast, playing, driving_amplitude1, driving_amplitude2, driving_frequency1, driving_frequency2
        if not playing:
            return
        if too_fast:
            return
        if not hasattr(self, '_t'):
            self._t = 0
        self._t += dt
        t = self._t
        # Use the shared integration function for a single step
        # Use float64 for GUI
        theta1, theta2, omega1, omega2, _, _ = integrate_double_pendulum(
            self.theta1, self.theta2, self.omega1, self.omega2,
            self.l1, self.l2, self.m1, self.m2, self.g,
            drag_coefficient1, drag_coefficient2,
            driving_amplitude1, driving_amplitude2,
            driving_frequency1, driving_frequency2,
            steps=1, dt=dt, device=None
        )
        self.theta1 = theta1.item() if hasattr(theta1, 'item') else float(theta1)
        self.theta2 = theta2.item() if hasattr(theta2, 'item') else float(theta2)
        self.omega1 = omega1.item() if hasattr(omega1, 'item') else float(omega1)
        self.omega2 = omega2.item() if hasattr(omega2, 'item') else float(omega2)
        # Detect excessive speed
        if abs(self.omega1) > FAST_OMEGA_THRESHOLD or abs(self.omega2) > FAST_OMEGA_THRESHOLD:
            too_fast = True
            playing = False
        global curr_step
        curr_step += 1

    def positions(self):
        x1 = self.l1 * math.sin(self.theta1)
        y1 = -self.l1 * math.cos(self.theta1)
        x2 = x1 + self.l2 * math.sin(self.theta2)
        y2 = y1 - self.l2 * math.cos(self.theta2)
        return x1, y1, x2, y2

class DoublePendulumApp:
    def __init__(self, root, pendulum):
        self.root = root
        self.pendulum = pendulum
        self.origin = initial_origin

        self.start_time = None

        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.pack()

        # Set a random target location in reachable area (same logic as train_controller)
        l1 = 1.0
        l2 = 1.0
        import numpy as np
        r = np.random.uniform(abs(l1 - l2), l1 + l2)
        theta = np.random.uniform(0, 2 * np.pi)
        self.target_x = r * np.cos(theta)
        self.target_y = -r * np.sin(theta)
        # Canvas item for the target box
        self.target_box = None
        self.final_box = None  # Add final_box attribute for red box

        global driving_amplitude1, driving_amplitude2, driving_frequency1, driving_frequency2, drag_coefficient1, drag_coefficient2
        driving_amplitude1 = trained_amp1
        driving_amplitude2 = trained_amp2
        driving_frequency1 = trained_freq1
        driving_frequency2 = trained_freq2
        drag_coefficient1 = trained_damp1
        drag_coefficient2 = trained_damp2
        # Also set initial positions globally if needed
        global initial_theta1, initial_theta2, initial_omega1, initial_omega2
        initial_theta1 = trained_theta1
        initial_theta2 = trained_theta2

        # Checkbox to enable/disable second mass using enable_second_mass function
        self.second_mass_enabled = tk.BooleanVar(value=True)
        self.second_mass_checkbox = tk.Checkbutton(
            self.root, text="Enable Second Mass", variable=self.second_mass_enabled,
            command=lambda: enable_second_mass(), bg="white", fg="black"
        )
        self.second_mass_checkbox.place(x=20, y=550)

        self.line1 = self.canvas.create_line(0, 0, 0, 0, width=5, fill="#00ff00")
        self.line2 = self.canvas.create_line(0, 0, 0, 0, width=5, fill="#00ff00")
        self.mass1 = self.canvas.create_oval(0, 0, 0, 0, fill="#00ff00")
        self.mass2 = self.canvas.create_oval(0, 0, 0, 0, fill="#00ff00")
        self.gravity_slider = tk.Scale(
            self.root, from_=0, to=-100, resolution=0.001, orient="horizontal", label="Gravity",
            length=300, showvalue=False, sliderlength=20, troughcolor="black", activebackground="red", bg="white", fg="black"
        )
        self.gravity_slider.set(gravity)
        self.gravity_slider.place(x=20, y=20)
        self.gravity_slider.bind("<Motion>", update_gravity)
        self._last_gravity = gravity
        # Damping sliders for each pendulum
        self.drag1_slider = tk.Scale(
            self.root, from_=-3, to=3, resolution=0.001, orient="horizontal", label="Damping 1",
            length=300, showvalue=True, sliderlength=20, troughcolor="black", activebackground="red", bg="white", fg="black"
        )
        self.drag1_slider.set(0.0)
        self.drag1_slider.place(x=20, y=80)
        self.drag1_slider.bind("<Motion>", update_drag1)
        self._last_drag1 = drag_coefficient1

        self.drag2_slider = tk.Scale(
            self.root, from_=-3, to=3, resolution=0.001, orient="horizontal", label="Damping 2",
            length=300, showvalue=True, sliderlength=20, troughcolor="black", activebackground="red", bg="white", fg="black"
        )
        self.drag2_slider.set(0.0)
        self.drag2_slider.place(x=20, y=140)
        self.drag2_slider.bind("<Motion>", update_drag2)
        self._last_drag2 = drag_coefficient2
        # Driving amplitude sliders
        self.amp1_slider = tk.Scale(
            self.root, from_=0, to=100, resolution=0.01, orient="horizontal", label="Driving Amplitude 1",
            length=300, showvalue=True, sliderlength=20, troughcolor="black", activebackground="red", bg="white", fg="black"
        )
        self.amp1_slider.set(driving_amplitude1)
        self.amp1_slider.place(x=20, y=200)
        self.amp1_slider.bind("<Motion>", update_amp1)
        self._last_amp1 = driving_amplitude1
        self.amp2_slider = tk.Scale(
            self.root, from_=0, to=100, resolution=0.01, orient="horizontal", label="Driving Amplitude 2",
            length=300, showvalue=True, sliderlength=20, troughcolor="black", activebackground="red", bg="white", fg="black"
        )
        self.amp2_slider.set(driving_amplitude2)
        self.amp2_slider.place(x=20, y=260)
        self.amp2_slider.bind("<Motion>", update_amp2)
        self._last_amp2 = driving_amplitude2

        self.freq1_slider = tk.Scale(
            self.root, from_=0, to=2, resolution=0.01, orient="horizontal", label="Driving Frequency 1",
            length=300, showvalue=True, sliderlength=20, troughcolor="black", activebackground="red", bg="white", fg="black"
        )
        self.freq1_slider.set(driving_frequency1)
        self.freq1_slider.place(x=20, y=320)
        self.freq1_slider.bind("<Motion>", update_freq1)
        self._last_freq1 = driving_frequency1

        self.freq2_slider = tk.Scale(
            self.root, from_=0, to=2, resolution=0.01, orient="horizontal", label="Driving Frequency 2",
            length=300, showvalue=True, sliderlength=20, troughcolor="black", activebackground="red", bg="white", fg="black"
        )
        self.freq2_slider.set(driving_frequency2)
        self.freq2_slider.place(x=20, y=380)
        self.freq2_slider.bind("<Motion>", update_freq2)
        self._last_freq2 = driving_frequency2
        

        # Overlay text for speed warning
        self.too_fast_text_id = self.canvas.create_text(
            500, 70, text="", fill="red", font=("Helvetica", 50, "bold")
        )

        self.update()
    def on_resize(self, event):
        self.canvas.config(width=event.width, height=event.height)
        self.origin = (event.width // 2 + x_origin_offset, event.height // 2 + y_origin_offset)

    def update(self):
        global gravity, drag_coefficient1, drag_coefficient2, driving_amplitude1, driving_amplitude2, driving_frequency1, driving_frequency2
        import time
        dt = 0.01
    # Use the current global parameters for the system
    # (If you want to update these from the optimizer, load them here)
        self.pendulum.step(dt)
        x1, y1, x2, y2 = self.pendulum.positions()
        x1_screen = self.origin[0] + x1 * 100
        y1_screen = self.origin[1] + y1 * 100
        x2_screen = self.origin[0] + x2 * 100
        y2_screen = self.origin[1] + y2 * 100

        # Draw the target location as a black box
        target_screen_x = self.origin[0] + target_x_global * 100
        target_screen_y = self.origin[1] + target_y_global * 100
        final_screen_x = self.origin[0] + final_x2 * 100
        final_screen_y = self.origin[1] + final_y2 * 100
        box_size = 10
        # Draw black box for target
        if not hasattr(self, 'target_box') or self.target_box is None:
            self.target_box = self.canvas.create_rectangle(
                float(target_screen_x) - box_size,
                float(target_screen_y) - box_size,
                float(target_screen_x) + box_size,
                float(target_screen_y) + box_size,
                fill="black"
            )
        else:
            self.canvas.coords(
                self.target_box,
                float(target_screen_x) - box_size, float(target_screen_y) - box_size,
                float(target_screen_x) + box_size, float(target_screen_y) + box_size
            )
        # Draw red box for final location
        if not hasattr(self, 'final_box') or self.final_box is None:
            self.final_box = self.canvas.create_rectangle(
                float(final_screen_x) - box_size,
                float(final_screen_y) - box_size,
                float(final_screen_x) + box_size,
                float(final_screen_y) + box_size,
                fill="red"
            )
        else:
            self.canvas.coords(
                self.final_box,
                float(final_screen_x) - box_size, float(final_screen_y) - box_size,
                float(final_screen_x) + box_size, float(final_screen_y) + box_size
            )

        # --- Sync sliders with their variables if changed externally ---

        # Gravity
        if self._last_gravity != gravity:
            self.gravity_slider.set(gravity)
            self._last_gravity = gravity
        else:
            self._last_gravity = self.gravity_slider.get()
        # Drag 1
        if self._last_drag1 != drag_coefficient1:
            self.drag1_slider.set(drag_coefficient1)
            self._last_drag1 = drag_coefficient1
        else:
            self._last_drag1 = self.drag1_slider.get()
        # Drag 2
        if self._last_drag2 != drag_coefficient2:
            self.drag2_slider.set(drag_coefficient2)
            self._last_drag2 = drag_coefficient2
        else:
            self._last_drag2 = self.drag2_slider.get()
        # Amp 1
        if self._last_amp1 != driving_amplitude1:
            self.amp1_slider.set(driving_amplitude1)
            self._last_amp1 = driving_amplitude1
        else:
            self._last_amp1 = self.amp1_slider.get()
        # Amp 2
        if self._last_amp2 != driving_amplitude2:
            self.amp2_slider.set(driving_amplitude2)
            self._last_amp2 = driving_amplitude2
        else:
            self._last_amp2 = self.amp2_slider.get()
        # Freq 1
        if self._last_freq1 != driving_frequency1:
            self.freq1_slider.set(driving_frequency1)
            self._last_freq1 = driving_frequency1
        else:
            self._last_freq1 = self.freq1_slider.get()
        # Freq 2
        if self._last_freq2 != driving_frequency2:
            self.freq2_slider.set(driving_frequency2)
            self._last_freq2 = driving_frequency2
        else:
            self._last_freq2 = self.freq2_slider.get()

        # Time for color animation and simulation looping
        if self.start_time is None:
            self.start_time = time.time()
        t = time.time() - self.start_time
        global curr_step
        if curr_step >= steps:
            curr_step = 0
            # Reset pendulum state, timer, and driving force time to specified initial values
            param_path = "optimized_params.npy"
            if os.path.exists(param_path):
                params = np.load(param_path)
                # Unpack all 10 parameters
                amp1, amp2, freq1, freq2, drag1, drag2, theta1_init, theta2_init, omega1_init, omega2_init = params
            else:
                theta1_init = 0.5  # default normalized value
                theta2_init = 0.5
                # ...existing code for other defaults...
            theta1_deg = theta1_init * 360 - 180
            theta2_deg = theta2_init * 360 - 180
            self.pendulum.m1 = 1.0
            self.pendulum.m2 = 1.0
            self.pendulum.l1 = 1.0
            self.pendulum.l2 = 1.0
            self.pendulum.theta1 = trained_theta1
            self.pendulum.theta2 = trained_theta2
            self.pendulum.omega1 = 0.0
            self.pendulum.omega2 = 0.0
            if hasattr(self.pendulum, '_t'):
                self.pendulum._t = 0.0
            self.start_time = time.time()
            global too_fast, playing
            too_fast = False
            playing = True

        # Calculate color for line1 (driving force 1)
        amp1 = driving_amplitude1
        freq1 = driving_frequency1
        # Color oscillates between green and blue/red, strength by amplitude
        # Use sine for smooth oscillation
        if freq1 > 0 and amp1 > 0:
            osc1 = math.sin(2 * math.pi * freq1 * t)
            # Clamp amplitude to [0, 1] for color scaling
            amp1_clamped = min(max(amp1 / 10.0, 0), 1)
            # Interpolate between green and blue/red
            # When osc1 > 0: blue, < 0: red, 0: green
            if osc1 > 0:
                # Green to blue
                blue = int(255 * amp1_clamped * osc1)
                color1 = f'#{0:02x}{int(255*(1-amp1_clamped*osc1)):02x}{blue:02x}'
            else:
                # Green to red
                red = int(255 * amp1_clamped * -osc1)
                color1 = f'#{red:02x}{int(255*(1-amp1_clamped*(-osc1))):02x}{0:02x}'
        else:
            color1 = "#00ff00"  # green

        # Calculate color for line2 (driving force 2)
        amp2 = driving_amplitude2
        freq2 = driving_frequency2
        if freq2 > 0 and amp2 > 0:
            osc2 = math.sin(2 * math.pi * freq2 * t)
            amp2_clamped = min(max(amp2 / 10.0, 0), 1)
            if osc2 > 0:
                blue = int(255 * amp2_clamped * osc2)
                color2 = f'#{0:02x}{int(255*(1-amp2_clamped*osc2)):02x}{blue:02x}'
            else:
                red = int(255 * amp2_clamped * -osc2)
                color2 = f'#{red:02x}{int(255*(1-amp2_clamped*(-osc2))):02x}{0:02x}'
        else:
            color2 = "#00ff00"

        self.canvas.coords(self.line1, self.origin[0], self.origin[1], x1_screen, y1_screen)
        self.canvas.coords(self.line2, x1_screen, y1_screen, x2_screen, y2_screen)

        # Animate mass1 color and size with driving force 1
        if freq1 > 0 and amp1 > 0:
            osc1 = math.sin(2 * math.pi * freq1 * t)
            amp1_clamped = min(max(amp1 / 10.0, 0), 1)
            # Color: blue/red/green oscillation
            if osc1 > 0:
                blue = int(255 * amp1_clamped * osc1)
                mass1_color = f'#{0:02x}{int(255*(1-amp1_clamped*osc1)):02x}{blue:02x}'
            else:
                red = int(255 * amp1_clamped * -osc1)
                mass1_color = f'#{red:02x}{int(255*(1-amp1_clamped*(-osc1))):02x}{0:02x}'
            # Size: base 20, add up to 10px with amplitude
            r1 = 20 + 10 * amp1_clamped * abs(osc1)
        else:
            mass1_color = "#0000ff"  # blue
            r1 = 20
        self.canvas.coords(self.mass1, x1_screen - r1, y1_screen - r1, x1_screen + r1, y1_screen + r1)

        # Animate mass2 color and size with driving force 2
        if freq2 > 0 and amp2 > 0:
            osc2 = math.sin(2 * math.pi * freq2 * t)
            amp2_clamped = min(max(amp2 / 10.0, 0), 1)
            if osc2 > 0:
                blue = int(255 * amp2_clamped * osc2)
                mass2_color = f'#{0:02x}{int(255*(1-amp2_clamped*osc2)):02x}{blue:02x}'
            else:
                red = int(255 * amp2_clamped * -osc2)
                mass2_color = f'#{red:02x}{int(255*(1-amp2_clamped*(-osc2))):02x}{0:02x}'
            r2 = 20 + 10 * amp2_clamped * abs(osc2)
        else:
            mass2_color = "#ff0000"  # red
            r2 = 20
        self.canvas.coords(self.mass2, x2_screen - r2, y2_screen - r2, x2_screen + r2, y2_screen + r2)

        # Set line colors
        self.canvas.itemconfig(self.line1, fill=color1)
        self.canvas.itemconfig(self.line2, fill=color2)

        # Update warning overlay visibility
        if too_fast:
            self.canvas.itemconfigure(self.too_fast_text_id, text="Too Fast!!!")
        else:
            self.canvas.itemconfigure(self.too_fast_text_id, text="")

        self.root.after(10, self.update)

def toggle_playing(event):
    global playing, pendulum_position, pendulum_speed, pendulum_angle, too_fast
    if too_fast and not playing:
        too_fast = False
    playing = not playing and not too_fast
    global mouse_has_pendulum
    mouse_has_pendulum = False

def grab(event):
    global playing, app
    x1, y1, x2, y2 = app.pendulum.positions()
    x1 *= 100
    y1 *= 100
    x2 *= 100
    y2 *= 100
    x1 += app.origin[0]
    y1 += app.origin[1]
    x2 += app.origin[0]
    y2 += app.origin[1]
    
    if (x1 - 10 < event.x < x1 + 10) and (y1 - 10 < event.y < y1 + 10):
        global mouse_has_pendulum_one
        mouse_has_pendulum_one = True
        playing = False
    elif (x2 - 10 < event.x < x2 + 10) and (y2 - 10 < event.y < y2 + 10):
        global mouse_has_pendulum_two
        mouse_has_pendulum_two = True
        playing = False
    else:
        mouse_has_pendulum_one = False
        mouse_has_pendulum_two = False

def drag(event):
    global mouse_has_pendulum_one, mouse_has_pendulum_two
    x1, y1, x2, y2 = app.pendulum.positions()
    x1 *= 100
    y1 *= 100
    x2 *= 100
    y2 *= 100
    if mouse_has_pendulum_one or mouse_has_pendulum_two:
        app.pendulum.omega1 = 0
        app.pendulum.omega2 = 0
    if mouse_has_pendulum_one:
        magnitude = math.sqrt((event.x - app.origin[0]) ** 2 + (event.y - app.origin[1]) ** 2)
        app.pendulum.l1 = magnitude/100
        app.pendulum.theta1 = math.atan2(event.x - app.origin[0], -event.y + app.origin[1])
    elif mouse_has_pendulum_two:
        magnitude = math.sqrt((event.x - x1 - app.origin[0]) ** 2 + (event.y - y1 - app.origin[1]) ** 2)
        app.pendulum.l2 = magnitude/100
        app.pendulum.theta2 = math.atan2(event.x - x1 - app.origin[0], -event.y + y1 + app.origin[1])
    else:
        return

def release(event):
    global mouse_has_pendulum_one, mouse_has_pendulum_two
    mouse_has_pendulum_one = False
    mouse_has_pendulum_two = False

def reset(event):
    global playing, app, too_fast
    app.pendulum.theta1 = math.radians(120)
    app.pendulum.theta2 = math.radians(90)
    app.pendulum.omega1 = 0
    app.pendulum.omega2 = 0
    app.pendulum.l1 = 1
    app.pendulum.l2 = 1
    too_fast = False
    playing = True
    
def update_gravity(event):
    global gravity
    gravity = app.gravity_slider.get()
    app.pendulum.g = gravity

def update_drag1(event):
    global drag_coefficient1
    drag_coefficient1 = app.drag1_slider.get()

def update_drag2(event):
    global drag_coefficient2
    drag_coefficient2 = app.drag2_slider.get()
    
def update_amp1(event):
    global driving_amplitude1
    driving_amplitude1 = app.amp1_slider.get()

def update_amp2(event):
    global driving_amplitude2
    driving_amplitude2 = app.amp2_slider.get()

def update_freq1(event):
    global driving_frequency1
    driving_frequency1 = app.freq1_slider.get()

def update_freq2(event):
    global driving_frequency2
    driving_frequency2 = app.freq2_slider.get()

def enable_second_mass():
    enabled = app.second_mass_enabled.get()
    if enabled:
        app.canvas.itemconfigure(app.mass2, state='normal')
        app.canvas.itemconfigure(app.line2, state='normal')
        app.pendulum.m2 = 1
    else:
        app.canvas.itemconfigure(app.mass2, state='hidden')
        app.canvas.itemconfigure(app.line2, state='hidden')
        app.pendulum.m2 = 0

if __name__ == "__main__":
    train_controller()

if __name__ == "__main__":
    global dp, app

    param_path = "optimized_params.npy"
    dp = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0, theta1=trained_theta1, theta2=trained_theta2)
    root = tk.Tk()
    app = DoublePendulumApp(root, dp)
    root.bind("<KeyPress-space>", toggle_playing)
    root.bind("<Button-1>", grab)
    root.bind("<B1-Motion>", drag)
    root.bind("<ButtonRelease-1>", release)
    root.bind("<KeyPress-r>", reset)
    root.mainloop()
