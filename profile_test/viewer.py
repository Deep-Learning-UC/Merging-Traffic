# tkinter window that shows frames live while main.py makes videos
import time

from PIL import Image, ImageTk


class Viewer:
    def __init__(self, title, fps=None):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="black")
        self.label = tk.Label(self.root, bg="black")
        self.label.pack()
        bar = tk.Frame(self.root, bg="black")
        bar.pack(fill="x")
        self.status = tk.Label(bar, text="", bg="black", fg="#9aa3b2", font=("DejaVu Sans", 11))
        self.status.pack(side="left", fill="x", expand=True)
        self.skip = False
        tk.Button(bar, text="Skip", command=self.press_skip).pack(side="right")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.open = True
        self.wait = 1 / fps if fps else 0
        self.last = 0

    # puts a frame on screen, waits if --realtime
    def show(self, frame, status=""):
        if not self.open:
            return
        photo = ImageTk.PhotoImage(Image.fromarray(frame))
        self.label.configure(image=photo)
        # if you dont keep a reference tkinter garbage collects the image and shows nothing. lost an hour to this
        self.label.image = photo
        self.status.configure(text=status)
        if self.wait:
            left = self.wait - (time.time() - self.last)
            if left > 0:
                time.sleep(left)
            self.last = time.time()
        # no mainloop, we just poke the window every frame. hacky but it works
        self.root.update()

    def press_skip(self):
        self.skip = True

    def close(self):
        if self.open:
            self.open = False
            self.root.destroy()


def make_viewer(title, fps=None):
    try:
        return Viewer(title, fps)
    except Exception as e:
        print(f"no tkinter window ({e}), still saving the videos")
        return None
