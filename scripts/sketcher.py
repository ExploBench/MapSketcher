import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import struct

class PointCloud3D:
    def __init__(self):
        self.points = np.empty((0, 3), dtype=float)
        self.colors = np.empty((0, 3), dtype=float)
        self.undo_stack = []
        self.cloud_height = 0.25
        self.export_layer_height = 0.25
        self.export_layer_count = 10

    def add_square(self, x, y, size=1.0, pts=200, color=(0.2,0.6,0.9)):
        pts = max(1, int(pts))
        n = int(np.sqrt(pts))
        if n < 1: n = 1
        xs = np.linspace(-size/2, size/2, n) + x
        ys = np.linspace(-size/2, size/2, n) + y
        xx, yy = np.meshgrid(xs, ys)
        xy = np.column_stack([xx.ravel(), yy.ravel()])
        z = np.zeros((len(xy),1), dtype=float)
        pts3 = np.column_stack([xy, z])
        cols = np.tile(np.array(color, dtype=float).reshape(1,3), (len(xy),1))
        start = len(self.points); count = len(pts3)
        self.undo_stack.append((start, count))
        if start == 0:
            self.points = pts3; self.colors = cols
        else:
            self.points = np.vstack([self.points, pts3])
            self.colors = np.vstack([self.colors, cols])
        return count

    def undo(self):
        if not self.undo_stack: return False
        start, count = self.undo_stack.pop()
        if start == 0:
            self.points = np.empty((0,3), dtype=float); self.colors = np.empty((0,3), dtype=float)
        else:
            self.points = self.points[:start].copy()
            self.colors = self.colors[:start].copy()
        return True

    def clear(self):
        if self.points.size == 0: return False
        self.undo_stack.append((0, len(self.points)))
        self.points = np.empty((0,3), dtype=float); self.colors = np.empty((0,3), dtype=float)
        return True

    def build_3d_layers(self):
        if self.points.size == 0: return None, None
        layers = max(1, int(self.export_layer_count)); step = abs(float(self.export_layer_height))
        base_pts = self.points; base_cols = (self.colors * 255).astype(np.uint8)
        pts_list, col_list = [], []
        for i in range(layers):
            layer = base_pts.copy(); layer[:,2] += i*step
            pts_list.append(layer); col_list.append(base_cols.copy())
        return np.vstack(pts_list), np.vstack(col_list)

    def save_ply(self, filename):
        pts, cols = self.build_3d_layers()
        if pts is None: return False
        with open(filename,"wb") as f:
            header = (
                f"ply\nformat binary_little_endian 1.0\n"
                f"element vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n"
            ); f.write(header.encode())
            for i in range(len(pts)):
                f.write(struct.pack("fff", float(pts[i,0]), float(pts[i,1]), float(pts[i,2])))
                f.write(struct.pack("BBB", int(cols[i,0]), int(cols[i,1]), int(cols[i,2])))
        return True

    def save_pcd(self, filename):
        pts, cols = self.build_3d_layers()
        if pts is None: return False
        with open(filename,"wb") as f:
            header = (
                "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\n"
                "SIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\n"
                f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
                f"POINTS {len(pts)}\nDATA binary\n"
            ); f.write(header.encode())
            for i in range(len(pts)):
                f.write(struct.pack("fff", float(pts[i,0]), float(pts[i,1]), float(pts[i,2])))
                rgb = (int(cols[i,0])<<16) | (int(cols[i,1])<<8) | int(cols[i,2])
                f.write(struct.pack("I", rgb))
        return True

class PointCloudApp:
    def __init__(self):
        self.pc = PointCloud3D()
        self.root = tk.Tk()
        self.root.title("Point Cloud - Debounced Redraw")
        self.root.geometry("1150x700")

        self.points_per_cloud = 200
        self.cloud_size = 1.0
        self.map_width = 20.0
        self.map_height = 15.0
        self.map_dpi = 100

        self.redraw_scheduled = False
        self.redraw_interval_ms = 30 

        self.scatter_obj = None

        self.mouse_down = False
        self.last_pos = None

        self.setup_ui()
        self.update_view(initial=True)

    def setup_ui(self):
        main = ttk.Frame(self.root, padding=10); main.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(main); left.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        right = ttk.Frame(main); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.build_settings(left); self.build_actions(left)

        self.fig = Figure(figsize=(8,6), dpi=self.map_dpi)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self.mouse_press)
        self.canvas.mpl_connect("button_release_event", self.mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.mouse_move)

    def build_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="Settings", padding=8); frame.pack(fill=tk.X)
        self.points_var = tk.IntVar(value=self.points_per_cloud)
        self.size_var = tk.DoubleVar(value=self.cloud_size)
        self.width_var = tk.DoubleVar(value=self.map_width)
        self.height_var = tk.DoubleVar(value=self.map_height)
        self.dpi_var = tk.IntVar(value=self.map_dpi)
        self.export_height_var = tk.DoubleVar(value=self.pc.export_layer_height)
        self.export_count_var = tk.IntVar(value=self.pc.export_layer_count)

        labels = [
            ("Points per cloud:", self.points_var),
            ("Cloud size (m):", self.size_var),
            ("Map width (m):", self.width_var),
            ("Map height (m):", self.height_var),
            ("Canvas DPI:", self.dpi_var),
            ("Layer height (m):", self.export_height_var),
            ("Layer count:", self.export_count_var),
        ]
        for i, (txt,var) in enumerate(labels):
            ttk.Label(frame,text=txt).grid(row=i,column=0,sticky=tk.W,pady=2)
            ttk.Entry(frame,textvariable=var,width=12).grid(row=i,column=1,pady=2)
        ttk.Button(frame,text="Apply",command=self.update_settings).grid(row=len(labels),column=0,columnspan=2,sticky="ew",pady=6)

    def build_actions(self, parent):
        frame = ttk.LabelFrame(parent, text="Actions", padding=8); frame.pack(fill=tk.X, pady=8)
        ttk.Button(frame, text="Undo", command=self.undo).pack(fill=tk.X, pady=3)
        ttk.Button(frame, text="Clear", command=self.clear).pack(fill=tk.X, pady=3)
        ttk.Button(frame, text="Save PLY", command=lambda: self.save("ply")).pack(fill=tk.X, pady=3)
        ttk.Button(frame, text="Save PCD", command=lambda: self.save("pcd")).pack(fill=tk.X, pady=3)
        ttk.Button(frame, text="Refresh", command=self.update_view).pack(fill=tk.X, pady=3)

    def update_settings(self):
        self.points_per_cloud = max(1, int(self.points_var.get()))
        self.cloud_size = float(self.size_var.get())
        self.map_width = float(self.width_var.get())
        self.map_height = float(self.height_var.get())
        self.map_dpi = max(30, min(800, int(self.dpi_var.get())))
        self.fig.set_dpi(self.map_dpi)
        self.pc.export_layer_height = abs(float(self.export_height_var.get()))
        self.pc.export_layer_count = max(1, int(self.export_count_var.get()))
        self.update_view()
        messagebox.showinfo("Settings", f"Export: {self.pc.export_layer_count} layers × {self.pc.export_layer_height} m")

    def mouse_press(self, event):
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self.mouse_down = True
            self.last_pos = (event.xdata, event.ydata)
            self.add_cloud_quick(event.xdata, event.ydata)

    def mouse_release(self, event):
        self.mouse_down = False
        self.last_pos = None

    def mouse_move(self, event):
        if not self.mouse_down or event.xdata is None: return
        x,y = event.xdata, event.ydata
        if self.last_pos is None:
            self.last_pos = (x,y); return
        lx,ly = self.last_pos
        if (x-lx)**2 + (y-ly)**2 > 0.0005:
            self.last_pos = (x,y)
            self.add_cloud_quick(x,y)

    def add_cloud_quick(self, x, y):
        """Add points to internal arrays quickly and schedule a redraw (debounced)."""
        self.pc.add_square(x, y, self.cloud_size, self.points_per_cloud)
        if not self.redraw_scheduled:
            self.redraw_scheduled = True
            self.root.after(self.redraw_interval_ms, self._perform_scheduled_redraw)

    def _perform_scheduled_redraw(self):
        self.redraw_scheduled = False
        self.update_view()

    def undo(self):
        if self.pc.undo():
            self.update_view()

    def clear(self):
        if messagebox.askyesno("Confirm", "Clear all points?"):
            if self.pc.clear():
                self.update_view()

    def save(self, fmt):
        if self.pc.points.size == 0:
            messagebox.showerror("Error","No points to save."); return
        path = filedialog.asksaveasfilename(defaultextension=f".{fmt}", filetypes=[(fmt.upper(), f"*.{fmt}")])
        if not path: return
        ok = self.pc.save_ply(path) if fmt=="ply" else self.pc.save_pcd(path)
        if ok:
            total_h = self.pc.export_layer_count * self.pc.export_layer_height
            messagebox.showinfo("Saved", f"Saved {self.pc.export_layer_count} layers × {self.pc.export_layer_height}m = {total_h:.3f}m")

    def update_view(self, initial=False):
        # downsample for display if many points, keep internal data intact
        MAX_DISPLAY = 30000  # adjust to taste
        total = len(self.pc.points)
        if total == 0:
            if self.scatter_obj is not None:
                self.scatter_obj.remove(); self.scatter_obj = None
            self.ax.set_xlim(0, self.map_width); self.ax.set_ylim(0, self.map_height); self.canvas.draw_idle()
            return

        if total <= MAX_DISPLAY:
            disp_idx = np.arange(total)
        else:
            disp_idx = np.linspace(0, total-1, MAX_DISPLAY).astype(int)

        pts2d = self.pc.points[disp_idx, :2]
        cols = self.pc.colors[disp_idx]

        self.ax.set_xlim(0, self.map_width); self.ax.set_ylim(0, self.map_height); self.ax.set_aspect('equal')

        if self.scatter_obj is None:
            self.scatter_obj = self.ax.scatter(pts2d[:,0], pts2d[:,1], c=cols, s=1, edgecolors='none')
        else:
            self.scatter_obj.set_offsets(pts2d)
            self.scatter_obj.set_facecolors(cols)

        # use draw_idle to allow backend to batch draws
        if initial:
            self.canvas.draw()
        else:
            self.canvas.draw_idle()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    PointCloudApp().run()
