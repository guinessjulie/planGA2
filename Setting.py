import configparser
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import ast  # 추가: 문자열을 리스트로 파싱하기 위한 모듈


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

        # 초기화 추가
        self.orientation_room_entries = []
        self.orientation_direction_entries = []

        self.size_room_entries = []
        self.size_room_entries = []
        self.min_size_entries = []
        self.max_size_entries = []

        self.room_name_entries = []
        self.room_number_entries = []

        self.adjacency_room1_entries = []
        self.adjacency_room2_entries = []
        self.adjacency_entries = []
        self.entry_frames = []  # 각 행의 프레임을 저장

        self.create_widgets()

    def create_widgets(self):
        self.sections = {}

        self.add_room_names_section()

        # OrientationRequirements 섹션을 통합
        self.add_orientation_requirements_section()
        self.add_size_requirements_section()
        self.add_adjacency_requirements_section()

        for section in self.config.sections():
            if section in ["OrientationRequirements", "SizeRequirements", "AdjacencyRequirements"]:
                continue  # 특정 섹션들을 건너뜁니다.

            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=section)

            self.sections[section] = {}

            for key, value in self.config.items(section):
                display_key = key.replace("num_rooms", "number of rooms")  # UI에 표시할 레이블 변경
                label = ttk.Label(frame, text=display_key)
                label.pack(side="top", fill="x", padx=10, pady=5)

                entry = ttk.Entry(frame)
                entry.insert(0, value)
                entry.pack(side="top", fill="x", padx=10, pady=5)

                self.sections[section][key] = entry

        save_button = ttk.Button(self.root, text="Save", command=self.save_settings)
        save_button.pack(pady=10)

    def add_room_names_section(self):
        # Room Names 섹션 생성
        section = "RoomNames"
        if not self.config.has_section(section):
            self.config.add_section(section)

        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Room Names")

        self.sections[section] = {}

        self.room_names_entries_frame = ttk.Frame(frame)
        self.room_names_entries_frame.pack(fill="x", padx=10, pady=5)

        # 컬럼 제목 추가
        header_frame = ttk.Frame(self.room_names_entries_frame)
        header_frame.pack(fill="x", pady=5)

        ttk.Label(header_frame, text="Room Number", width=20).pack(side="left", padx=5)
        ttk.Label(header_frame, text="Room Name", width=20).pack(side="left", padx=5)

        # 기존 항목들을 가로로 배치
        for key, value in self.config.items(section):
            self.add_existing_room_name_entry(key, value)

        # 방 번호와 이름 추가 버튼
        add_button = ttk.Button(frame, text="Add Room Name", command=self.add_room_name_entry)
        add_button.pack(pady=10)

    def add_existing_room_name_entry(self, room_number, room_name):
        self.add_room_name_entry(room_number, room_name)

    def add_room_name_entry(self, room_number="", room_name=""):
        row_frame = ttk.Frame(self.room_names_entries_frame)
        row_frame.pack(fill="x", pady=5)

        room_number_entry = ttk.Entry(row_frame, width=20)
        room_number_entry.insert(0, room_number)
        room_number_entry.pack(side="left", padx=5)
        self.room_number_entries.append(room_number_entry)

        room_name_entry = ttk.Entry(row_frame, width=20)
        room_name_entry.insert(0, room_name)
        room_name_entry.pack(side="left", padx=5)
        self.room_name_entries.append(room_name_entry)

        # 제거 버튼 추가
        remove_button = ttk.Button(row_frame, text="Remove",
                                   command=lambda: self.remove_room_name_entry(row_frame, room_number_entry,
                                                                               room_name_entry))
        remove_button.pack(side="left", padx=5)

        self.entry_frames.append(row_frame)

    def remove_room_name_entry(self, row_frame, room_number_entry, room_name_entry):
        # 설정 파일에서 해당 항목 삭제
        section = "RoomNames"
        room_number = room_number_entry.get()
        if room_number in self.config[section]:
            del self.config[section][room_number]

        # 행 프레임과 해당 입력 필드들 제거
        row_frame.pack_forget()
        self.room_number_entries.remove(room_number_entry)
        self.room_name_entries.remove(room_name_entry)
        self.entry_frames.remove(row_frame)

    def add_orientation_requirements_section(self):
        # OrientationRequirements 섹션 생성
        section = "OrientationRequirements"
        if not self.config.has_section(section):
            self.config.add_section(section)

        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text='Orientation')

        self.sections[section] = {}

        self.orientation_entries_frame = ttk.Frame(frame)
        self.orientation_entries_frame.pack(fill="x", padx=10, pady=5)

        # 컬럼 제목 추가
        header_frame = ttk.Frame(self.orientation_entries_frame)
        header_frame.pack(fill="x", pady=5)

        ttk.Label(header_frame, text="Room Number", width=20).pack(side="left", padx=5)
        ttk.Label(header_frame, text="Direction", width=20).pack(side="left", padx=5)

        # 기존 항목들을 가로로 배치
        for key, value in self.config.items(section):
            self.add_existing_orientation_room_entry(key, value)

        # 방 번호와 방향 추가 버튼
        add_button = ttk.Button(frame, text="Add Room", command=self.add_orientation_room_entry)
        add_button.pack(pady=10)

    def add_size_requirements_section(self):
        # SizeRequirements 섹션 생성
        section = "SizeRequirements"
        if not self.config.has_section(section):
            self.config.add_section(section)

        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text='Size')

        self.sections[section] = {}

        self.size_entries_frame = ttk.Frame(frame)
        self.size_entries_frame.pack(fill="x", padx=10, pady=5)

        # 컬럼 제목 추가
        header_frame = ttk.Frame(self.size_entries_frame)
        header_frame.pack(fill="x", pady=5)

        ttk.Label(header_frame, text="Room Number", width=20).pack(side="left", padx=5)
        ttk.Label(header_frame, text="Min Size", width=20).pack(side="left", padx=5)
        ttk.Label(header_frame, text="Max Size", width=20).pack(side="left", padx=5)

        # 기존 항목들을 가로로 배치
        for key, value in self.config.items(section):
            self.add_existing_size_room_entry(key, value)

        # 방 번호와 크기 추가 버튼
        add_button = ttk.Button(frame, text="Add Room", command=self.add_size_room_entry)
        add_button.pack(pady=10)

    def add_adjacency_requirements_section(self):
        # AdjacencyRequirements 섹션 생성
        section = "AdjacencyRequirements"
        if not self.config.has_section(section):
            self.config.add_section(section)

        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text='Adjacency')

        self.sections[section] = {}

        self.adjacency_entries_frame = ttk.Frame(frame)
        self.adjacency_entries_frame.pack(fill="x", padx=10, pady=5)

        # 컬럼 제목 추가
        header_frame = ttk.Frame(self.adjacency_entries_frame)
        header_frame.pack(fill="x", pady=5)

        ttk.Label(header_frame, text="Room 1", width=20).pack(side="left", padx=5)
        ttk.Label(header_frame, text="Room 2", width=20).pack(side="left", padx=5)

        # 기존 항목들을 가로로 배치
        if self.config.has_option(section, "adjacent"):
            adjacency_list = ast.literal_eval(self.config.get(section, "adjacent"))
            for room1, room2 in adjacency_list:
                self.add_existing_adjacency_entry(room1, room2)

        # 방 번호와 인접 관계 추가 버튼
        add_button = ttk.Button(frame, text="Add Adjacency", command=self.add_adjacency_entry)
        add_button.pack(pady=10)

    def add_existing_orientation_room_entry(self, room_number, direction):
        self.add_orientation_room_entry(room_number, direction)

    def add_orientation_room_entry(self, room_number="", direction=""):
        row_frame = ttk.Frame(self.orientation_entries_frame)
        row_frame.pack(fill="x", pady=5)

        room_entry = ttk.Entry(row_frame, width=20)
        room_entry.insert(0, room_number)
        room_entry.pack(side="left", padx=5)
        self.orientation_room_entries.append(room_entry)

        direction_combobox = ttk.Combobox(row_frame, values=["south", "north", "west", "east"], width=18)
        direction_combobox.set(direction)
        direction_combobox.pack(side="left", padx=5)
        self.orientation_direction_entries.append(direction_combobox)

        # 제거 버튼 추가
        remove_button = ttk.Button(row_frame, text="Remove", command=lambda: self.remove_orientation_room_entry(row_frame, room_entry, direction_combobox))
        remove_button.pack(side="left", padx=5)

        self.entry_frames.append(row_frame)

    def add_existing_size_room_entry(self, room_number, size_range):
        min_size, max_size = size_range.split(',')
        self.add_size_room_entry(room_number, min_size, max_size)

    def add_size_room_entry(self, room_number="", min_size="", max_size=""):
        row_frame = ttk.Frame(self.size_entries_frame)
        row_frame.pack(fill="x", pady=5)

        room_entry = ttk.Entry(row_frame, width=20)
        room_entry.insert(0, room_number)
        room_entry.pack(side="left", padx=5)
        self.size_room_entries.append(room_entry)

        min_size_entry = ttk.Entry(row_frame, width=20)
        min_size_entry.insert(0, min_size)
        min_size_entry.pack(side="left", padx=5)
        self.min_size_entries.append(min_size_entry)

        min_size_label = ttk.Label(row_frame, text="m²")
        min_size_label.pack(side="left", padx=5)

        max_size_entry = ttk.Entry(row_frame, width=20)
        max_size_entry.insert(0, max_size)
        max_size_entry.pack(side="left", padx=5)
        self.max_size_entries.append(max_size_entry)

        max_size_label = ttk.Label(row_frame, text="m²")
        max_size_label.pack(side="left", padx=5)

        # 제거 버튼 추가
        remove_button = ttk.Button(row_frame, text="Remove",
                                   command=lambda: self.remove_size_room_entry(row_frame, room_entry, min_size_entry,
                                                                               max_size_entry))
        remove_button.pack(side="left", padx=5)

        self.entry_frames.append(row_frame)

    def add_existing_adjacency_entry(self, room1, room2):
        self.add_adjacency_entry(room1, room2)

    def add_adjacency_entry(self, room1="", room2=""):
        row_frame = ttk.Frame(self.adjacency_entries_frame)
        row_frame.pack(fill="x", pady=5)

        room1_entry = ttk.Entry(row_frame, width=20)
        room1_entry.insert(0, room1)
        room1_entry.pack(side="left", padx=5)
        self.adjacency_room1_entries.append(room1_entry)

        room2_entry = ttk.Entry(row_frame, width=20)
        room2_entry.insert(0, room2)
        room2_entry.pack(side="left", padx=5)
        self.adjacency_room2_entries.append(room2_entry)

        # 제거 버튼 추가
        remove_button = ttk.Button(row_frame, text="Remove", command=lambda: self.remove_adjacency_entry(row_frame, room1_entry, room2_entry))
        remove_button.pack(side="left", padx=5)

        self.entry_frames.append(row_frame)

    def remove_orientation_room_entry(self, row_frame, room_entry, direction_combobox):
        # 설정 파일에서 해당 항목 삭제
        section = "OrientationRequirements"
        room_number = room_entry.get()
        if room_number in self.config[section]:
            del self.config[section][room_number]

        # 행 프레임과 해당 입력 필드들 제거
        row_frame.pack_forget()
        self.orientation_room_entries.remove(room_entry)
        self.orientation_direction_entries.remove(direction_combobox)
        self.entry_frames.remove(row_frame)

    def remove_size_room_entry(self, row_frame, room_entry, size_combobox):
        # 설정 파일에서 해당 항목 삭제
        section = "SizeRequirements"
        room_number = room_entry.get()
        if room_number in self.config[section]:
            del self.config[section][room_number]

        # 행 프레임과 해당 입력 필드들 제거
        row_frame.pack_forget()
        self.size_room_entries.remove(room_entry)
        self.size_entries.remove(size_combobox)
        self.entry_frames.remove(row_frame)

    def remove_adjacency_entry(self, row_frame, room1_entry, room2_entry):
        # 설정 파일에서 해당 항목 삭제
        section = "AdjacencyRequirements"
        room1 = room1_entry.get()
        room2 = room2_entry.get()
        adjacency_list = ast.literal_eval(self.config.get(section, "adjacent"))
        adjacency_list.remove((int(room1), int(room2)))
        self.config.set(section, "adjacent", str(adjacency_list))

        # 행 프레임과 해당 입력 필드들 제거
        row_frame.pack_forget()
        self.adjacency_room1_entries.remove(room1_entry)
        self.adjacency_room2_entries.remove(room2_entry)
        self.entry_frames.remove(row_frame)

    def save_settings(self):
        for section, entries in self.sections.items():
            for key, entry in entries.items():
                self.config[section][key] = entry.get()

        # OrientationRequirements 섹션 저장
        section = "OrientationRequirements"
        for i, (room_entry, direction_combobox) in enumerate(
                zip(self.orientation_room_entries, self.orientation_direction_entries)):
            room_number = room_entry.get()
            direction = direction_combobox.get()
            if room_number and direction:
                self.config[section][room_number] = direction

        # SizeRequirements 섹션 저장
        section = "SizeRequirements"
        for i, (room_entry, min_size_entry, max_size_entry) in enumerate(
                zip(self.size_room_entries, self.min_size_entries, self.max_size_entries)):
            room_number = room_entry.get()
            min_size = min_size_entry.get()
            max_size = max_size_entry.get()
            if room_number and min_size and max_size:
                self.config[section][room_number] = f"{min_size},{max_size}"

        # AdjacencyRequirements 섹션 저장
        section = "AdjacencyRequirements"
        adjacency_list = []
        for i, (room1_entry, room2_entry) in enumerate(zip(self.adjacency_room1_entries, self.adjacency_room2_entries)):
            room1 = room1_entry.get()
            room2 = room2_entry.get()
            if room1 and room2:
                adjacency_list.append((int(room1), int(room2)))
        self.config.set(section, "adjacent", str(adjacency_list))

        # RoomNames 섹션 저장
        section = "RoomNames"
        for i, (room_number_entry, room_name_entry) in enumerate(zip(self.room_number_entries, self.room_name_entries)):
            room_number = room_number_entry.get()
            room_name = room_name_entry.get()
            if room_number and room_name:
                self.config[section][room_number] = room_name

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

if __name__ == "__main__":
    main()