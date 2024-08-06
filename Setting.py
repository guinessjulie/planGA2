import configparser
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


def read_ini_file(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config


def write_ini_file(config, file_name):
    with open(file_name, 'w') as configfile:
        config.write(configfile)


class SettingsApp:
    def __init__(self, root, config):
        self.config = config
        self.root = root
        self.root.title("Settings")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.create_widgets()

    def create_widgets(self):
        self.sections = {}

        for section in self.config.sections():
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=section)

            self.sections[section] = {}

            for key, value in self.config.items(section):
                label = ttk.Label(frame, text=key)
                label.pack(side="top", fill="x", padx=10, pady=5)

                entry = ttk.Entry(frame)
                entry.insert(0, value)
                entry.pack(side="top", fill="x", padx=10, pady=5)

                self.sections[section][key] = entry

        save_button = ttk.Button(self.root, text="Save", command=self.save_settings)
        save_button.pack(pady=10)

    def save_settings(self):
        for section, entries in self.sections.items():
            for key, entry in entries.items():
                self.config[section][key] = entry.get()

        write_ini_file(self.config, 'constraints.ini')
        messagebox.showinfo("Settings", "Settings saved successfully!")


def main():
    root = tk.Tk()
    root.title("Main Application")

    def open_settings():
        settings_root = tk.Toplevel(root)
        settings_root.title("Settings")
        config = read_ini_file('constraints.ini')
        SettingsApp(settings_root, config)

    main_frame = ttk.Frame(root)
    main_frame.pack(padx=10, pady=10)

    button1 = ttk.Button(main_frame, text="Button 1")
    button1.pack(pady=5)

    button2 = ttk.Button(main_frame, text="Button 2")
    button2.pack(pady=5)

    settings_button = ttk.Button(main_frame, text="Settings", command=open_settings)
    settings_button.pack(pady=5)

    root.mainloop()

# if __name__ == '__main__':
#    main()
