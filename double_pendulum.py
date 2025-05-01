import math
import tkinter as tk

mouse_has_pendulum_one = False
mouse_has_pendulum_two = False
playing = True
initial_origin = (400, 300)
gravity = -9.81
drag_coefficient = 0

class DoublePendulum:
    def __init__(self, m1, m2, l1, l2, theta1, theta2, g=gravity):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2 
        self.g = g
        

        self.theta1 = math.radians(theta1)
        self.theta2 = math.radians(theta2)
        self.omega1 = 0
        self.omega2 = 0

    def step(self, dt):
        if not playing:
            return
        delta = self.theta2 - self.theta1

        denom1 = (self.m1 + self.m2) * self.l1 - self.m2 * self.l1 * math.cos(delta) ** 2
        denom2 = (self.l2 / self.l1) * denom1

        domega1_dt = ((self.m2 * self.l1 * self.omega1 ** 2 * math.sin(delta) * math.cos(delta) +
                       self.m2 * self.g * math.sin(self.theta2) * math.cos(delta) +
                       self.m2 * self.l2 * self.omega2 ** 2 * math.sin(delta) -
                       (self.m1 + self.m2) * self.g * math.sin(self.theta1)) / denom1) - drag_coefficient * self.omega1

        domega2_dt = ((-self.m2 * self.l2 * self.omega2 ** 2 * math.sin(delta) * math.cos(delta) +
                       (self.m1 + self.m2) * self.g * math.sin(self.theta1) * math.cos(delta) -
                       (self.m1 + self.m2) * self.l1 * self.omega1 ** 2 * math.sin(delta) -
                       (self.m1 + self.m2) * self.g * math.sin(self.theta2)) / denom2) - drag_coefficient * self.omega2

        self.omega1 += domega1_dt * dt
        self.omega2 += domega2_dt * dt
        self.theta1 += self.omega1 * dt
        self.theta2 += self.omega2 * dt

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

        self.canvas = tk.Canvas(root, width=800, height=600, bg="white")
        self.canvas.pack()

        self.line1 = self.canvas.create_line(0, 0, 0, 0, width=2, fill="blue")
        self.line2 = self.canvas.create_line(0, 0, 0, 0, width=2, fill="red")
        self.mass1 = self.canvas.create_oval(0, 0, 0, 0, fill="blue")
        self.mass2 = self.canvas.create_oval(0, 0, 0, 0, fill="red")
        self.gravity_slider = tk.Scale(
            self.root, from_=0, to=-100, resolution=0.001, orient="horizontal", label="Gravity",
            length=300, showvalue=0, sliderlength=20, troughcolor="black", activebackground="red", bg="white", fg="black"
        )
        
        self.gravity_slider.set(gravity)
        self.gravity_slider.place(x=20, y=20)
        self.gravity_slider.bind("<Motion>", update_gravity)
        
        self.drag_slider = tk.Scale(
            self.root, from_=0, to=3, resolution=0.001, orient="horizontal", label="Drag",
            length=300, showvalue=0, sliderlength=20, troughcolor="black", activebackground="red", bg="white", fg="black"
        )
        
        self.drag_slider.set(0)
        self.drag_slider.place(x=20, y=80)
        self.drag_slider.bind("<Motion>", update_drag)

        self.update()

    def update(self):
        dt = 0.01
        self.pendulum.step(dt)
        x1, y1, x2, y2 = self.pendulum.positions()
        x1_screen = self.origin[0] + x1 * 100
        y1_screen = self.origin[1] + y1 * 100
        x2_screen = self.origin[0] + x2 * 100
        y2_screen = self.origin[1] + y2 * 100

        self.canvas.coords(self.line1, self.origin[0], self.origin[1], x1_screen, y1_screen)
        self.canvas.coords(self.line2, x1_screen, y1_screen, x2_screen, y2_screen)
        self.canvas.coords(self.mass1, x1_screen - 10, y1_screen - 10, x1_screen + 10, y1_screen + 10)
        self.canvas.coords(self.mass2, x2_screen - 10, y2_screen - 10, x2_screen + 10, y2_screen + 10)

        self.root.after(10, self.update)

def toggle_playing(event):
    global playing, pendulum_position, pendulum_speed, pendulum_angle
    playing = not playing
    global mouse_has_pendulum
    mouse_has_pendulum = False

def grab(event):
    global playing, app
    x1, y1, x2, y2 = app.pendulum.positions()
    x1 *= 100
    y1 *= 100
    x2 *= 100
    y2 *= 100
    x1 += initial_origin[0]
    y1 += initial_origin[1]
    x2 += initial_origin[0]
    y2 += initial_origin[1]
    
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
    global playing, app
    app.pendulum.theta1 = math.radians(120)
    app.pendulum.theta2 = math.radians(90)
    app.pendulum.omega1 = 0
    app.pendulum.omega2 = 0
    app.pendulum.l1 = 1
    app.pendulum.l2 = 1
    
def update_gravity(event):
    global gravity
    gravity = app.gravity_slider.get()
    app.pendulum.g = gravity

def update_drag(event):
    global drag_coefficient
    drag_coefficient = app.drag_slider.get()
    app.pendulum.drag = drag_coefficient

if __name__ == "__main__":
    global dp, app

    dp = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0, theta1=120, theta2=90)
    root = tk.Tk()
    app = DoublePendulumApp(root, dp)
    root.bind("<KeyPress-space>", toggle_playing)
    root.bind("<Button-1>", grab)
    root.bind("<B1-Motion>", drag)
    root.bind("<ButtonRelease-1>", release)
    root.bind("<KeyPress-r>", reset)
    root.mainloop()