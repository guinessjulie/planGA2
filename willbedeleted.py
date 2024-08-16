import tkinter as tk
from tkinter import ttk
# todo 1. show simplify
# todo 2. draw plan equal thickness show option
class SettingsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Main Application")

        self.create_widgets()

    def create_widgets(self):
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        button_width = 20

        buttons = [
            ("Create Initial Floorplan", self.initialize_floorplan),
            ("Draw Floorplan", self.draw_floorplan),
            ("Simplify Floorplan", self.exchange_cells),
            ("Draw Plan Equal Thickness", self.draw_equal_thickness),
            ("Return Floorplan", self.return_floorplan),
            ("Exit", self.root.quit),
        ]

        for text, command in buttons:
            tk.Button(left_frame, text=text, command=command, width=button_width).pack(pady=5, padx=5, fill=tk.X)

        self.canvas = tk.Canvas(right_frame, width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def initialize_floorplan(self):
        # Placeholder for actual functionality
        pass

    def draw_floorplan(self):
        # Placeholder for actual functionality
        pass

    def exchange_cells(self):
        # Placeholder for actual functionality
        pass

    def draw_equal_thickness(self):
        # Placeholder for actual functionality
        pass

    def return_floorplan(self):
        # Placeholder for actual functionality
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = SettingsApp(root)
    root.mainloop()
