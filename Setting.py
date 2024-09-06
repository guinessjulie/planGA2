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
    def __init__(self, root, config, constraints):
        self.config = config
        self.constraints = constraints
        self.root = root
        self.root.title("Settings")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        # 초기화 추가

        self.room_names = self.load_room_names()
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
        # 스타일 설정
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook.Tab", padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", "#FFCCCC")], foreground=[("selected", "#000000")])

        # 일반 프레임의 배경색 설정
        style.configure("TFrame", background="#E6E6E6")

        # 라벨 스타일 변경
        style.configure("TLabel", background="#E6E6E6", )

        # 버튼 스타일 변경
        style.configure("TButton", padding=[4,2], relief="raised", background="#CCCCFF")
        style.map("TButton", background=[("active", "#FF99CC")], foreground=[("disabled", "#B0B0B0")])


        self.sections = {}

        self.add_room_names_section()

        # OrientationRequirements 섹션을 통합
        self.add_orientation_requirements_section()
        self.add_size_requirements_section()
        self.add_adjacency_requirements_section()

        for section in self.config.sections():
            if section in ["RoomNames"]:
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

    def update_room_number_label(self, room_name_combobox, room_number_label):
        """ 방 이름이 선택되면 방 번호를 업데이트 """
        selected_room_name = room_name_combobox.get()
        for room_number, room_name in self.room_names.items():
            if room_name == selected_room_name:
                room_number_label.config(text=str(room_number))
                break


##################################################################
# info: Room Name
##################################################################
    def load_room_names(self):
        room_names = {}
        if self.config.has_section('RoomNames'):
            room_names = {int(key): value for key, value in self.config.items("RoomNames")}
        return room_names

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

        ttk.Label(header_frame, text="ID", width=5).pack(side="left", padx=5)
        ttk.Label(header_frame, text="Room Name", width=20).pack(side="left", padx=5)

        # 기존 항목들을 가로로 배치
        for key, value in self.config.items(section):
            self.add_existing_room_name_entry(key, value)

        # 방 번호와 이름 추가 버튼
        add_button = ttk.Button(frame, text="Add Room Name", command=self.add_room_name_entry)
        add_button.pack(pady=10)

    def add_existing_room_name_entry(self, room_number, room_name):
        self.add_room_name_entry(room_number, room_name)

    # info 방 이름이 입력되었을 때 해당 값을 SettingApp에 제대로 전달하려면, Entry 위젯에서 발생하는 이벤트를 처리해야 함.
    #  특히, 사용자가 Entry 위젯에 값을 입력한 후 그 값을 즉시 반영할 수 있도록 이벤트 핸들러 추가
    def add_room_name_entry(self, room_number=None, room_name=""):
        row_frame = ttk.Frame(self.room_names_entries_frame)
        row_frame.pack(fill="x", pady=5)

        if room_number is None:
            # 현재 사용 중인 방 번호를 추출
            existing_numbers = sorted(int(label.cget("text")) for label in self.room_number_entries)

            # 가장 작은 빈 번호를 찾음
            room_number = 1
            for number in existing_numbers:
                if room_number < number:
                    break
                room_number += 1

        # 방 번호를 Label로 표시
        room_number_label = ttk.Label(row_frame, text=str(room_number), width=5)
        room_number_label.pack(side="left", padx=5)
        self.room_number_entries.append(room_number_label)

        # 방 이름을 입력할 수 있는 Entry
        # info StringVar 사용:
        #  StringVar는 값이 변경될 때 이벤트를 트리거할 수 있음.
        room_name_var = tk.StringVar(value=room_name)
        room_name_entry = ttk.Entry(row_frame, textvariable=room_name_var, width=20)
        room_name_entry.pack(side="left", padx=5)
        self.room_name_entries.append(room_name_entry)

        # 방 이름 변경 시 이벤트 처리 (이벤트 트리거가 아닌, 저장 시점에 값을 가져오도록 변경)
        room_name_entry.bind("<FocusOut>", lambda event: self.update_room_name(room_number_label, room_name_var))

        # 제거 버튼 추가
        remove_button = ttk.Button(row_frame, text="Remove",
                                   command=lambda: self.remove_room_name_entry(row_frame, room_number_label,
                                                                               room_name_entry))
        remove_button.pack(side="left", padx=5)

        self.entry_frames.append(row_frame)

    def remove_room_name_entry(self, row_frame, room_number_label, room_name_entry):
        # 설정 파일에서 해당 항목 삭제
        section = "RoomNames"
        room_number = room_number_label.cget("text")
        if room_number in self.config[section]:
            del self.config[section][room_number]

        # 행 프레임과 해당 입력 필드들 제거
        row_frame.pack_forget()
        self.room_number_entries.remove(room_number_label)
        self.room_name_entries.remove(room_name_entry)
        self.entry_frames.remove(row_frame)

        room_number_label.destroy()  # Label 제거
        room_name_entry.destroy()  # Entry 제거

    def save_room_names(self):
        section = "RoomNames"
        if not self.config.has_section(section):
            self.config.add_section(section)

        # 기존 섹션을 모두 비우고 새로 작성
        self.config.remove_section(section)
        self.config.add_section(section)

        for room_number_label, room_name_entry in zip(self.room_number_entries, self.room_name_entries):
            room_number = room_number_label.cget("text")  # Label에서 방 번호를 가져옴
            room_name = room_name_entry.get().strip()  # 현재 입력된 방 이름을 가져옴
            if room_number and room_name:
                self.config.set(section, room_number, room_name)
            else:
                if room_number in self.config[section]:
                    del self.config[section][room_number]

        write_ini_file(self.config, 'config.ini')

    ############################################################
    # info: orientation_requirements_section
    ############################################################

    def add_orientation_requirements_section(self):
        # OrientationRequirements 섹션 생성
        section = "OrientationRequirements"
        if not self.constraints.has_section(section):
            self.constraints.add_section(section)

        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text='Orientation')

        self.sections[section] = {}

        self.orientation_entries_frame = ttk.Frame(frame)
        self.orientation_entries_frame.pack(fill="x", padx=10, pady=5)

        # 컬럼 제목 추가
        header_frame = ttk.Frame(self.orientation_entries_frame)
        header_frame.pack(fill="x", pady=5)

        # 컬럼 헤더들
        ttk.Label(header_frame, text="ID", width=5).pack(side="left", padx=5)
        ttk.Label(header_frame, text="Room Name", width=20).pack(side="left", padx=5)
        ttk.Label(header_frame, text="Direction", width=15).pack(side="left", padx=5)
        ttk.Label(header_frame, text="", width=5).pack(side="left", padx=5)  # Remove 버튼 위치를 위한 빈 라벨

        # 기존 항목들을 추가
        for key, value in self.constraints.items(section):
            self.add_existing_orientation_room_entry(key, value)

        # 방 번호와 방향 추가 버튼
        add_button = ttk.Button(frame, text="Add Room", command=self.add_orientation_room_entry)
        add_button.pack(pady=10)

    def add_existing_orientation_room_entry(self, room_number, direction):
        room_number = int(room_number)
        room_name = self.room_names.get(room_number, 'Unknown')
        self.add_orientation_room_entry(room_number, room_name, direction)

    def add_orientation_room_entry(self, room_number=None, room_name="Unknown", direction=None):
        row_frame = ttk.Frame(self.orientation_entries_frame)
        row_frame.pack(fill="x", pady=5)

        # 방 번호를 read-only Label로 표시
        room_number_label = ttk.Label(row_frame, text=str(room_number) if room_number else "Unknown", width=5)
        room_number_label.pack(side="left", padx=5)

        # 방 이름을 선택할 수 있는 Combobox
        room_name_var = tk.StringVar(value=room_name)
        room_name_combobox = ttk.Combobox(row_frame, textvariable=room_name_var, width=20, state="readonly")
        room_name_combobox['values'] = list(self.room_names.values())  # room_names 딕셔너리의 values를 나열
        room_name_combobox.pack(side="left", padx=5)

        # Combobox가 초기화된 후에 값을 설정
        self.orientation_entries_frame.after(100, lambda: room_name_combobox.set(room_name))

        # 방 이름 선택 시 해당 방 번호를 Label에 업데이트
        room_name_combobox.bind("<<ComboboxSelected>>",
                                lambda e: self.update_room_number_label(room_name_combobox, room_number_label))

        # Direction을 표시하는 Combobox
        direction_var = tk.StringVar(value=direction if direction else "")
        direction_combobox = ttk.Combobox(row_frame, textvariable=direction_var, width=15)
        direction_combobox['values'] = ("north", "south", "east", "west")  # 가능한 방향 목록
        direction_combobox.pack(side="left", padx=5)

        if direction:
            self.orientation_entries_frame.after(100, lambda: direction_combobox.set(direction))

        remove_button = ttk.Button(row_frame, text="Remove",
                                   command=lambda: self.remove_orientation_room_entry(row_frame, room_number_label,
                                                                                      direction_combobox))
        remove_button.pack(side="left", padx=5)

        self.orientation_room_entries.append(room_number_label)
        self.orientation_direction_entries.append(direction_combobox)
        self.entry_frames.append(row_frame)

    def remove_orientation_room_entry(self, row_frame, room_number_label, direction_combobox):
        # 설정 파일에서 해당 항목 삭제
        section = "OrientationRequirements"
        room_number = room_number_label.cget("text")  # Label에서 방 번호를 가져옴
        if room_number in self.constraints[section]:
            del self.constraints[section][room_number]

        # 행 프레임과 해당 입력 필드들 제거
        row_frame.pack_forget()
        self.orientation_room_entries.remove(room_number_label)
        self.orientation_direction_entries.remove(direction_combobox)
        self.entry_frames.remove(row_frame)

    def save_orientation_requirements(self):
        section = "OrientationRequirements"
        for i, (room_label, direction_combobox) in enumerate(
                zip(self.orientation_room_entries, self.orientation_direction_entries)):
            room_number = room_label.cget("text")  # Label에서 방 번호를 가져옴
            direction = direction_combobox.get()
            if room_number and direction:
                self.constraints[section][room_number] = direction

    def set_direction(self, combobox, direction):
        if direction:
            combobox.set(direction)

    ############################################################
    # info: size_requirements_section
    ############################################################

    def add_size_requirements_section(self):
        # SizeRequirements 섹션 생성
        section = "SizeRequirements"
        if not self.constraints.has_section(section):
            self.constraints.add_section(section)

        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text='Size')

        self.sections[section] = {}

        self.size_entries_frame = ttk.Frame(frame)
        self.size_entries_frame.pack(fill="x", padx=10, pady=5)

        # 컬럼 제목 추가
        header_frame = ttk.Frame(self.size_entries_frame)
        header_frame.pack(fill="x", pady=5)

        ttk.Label(header_frame, text="ID", width=5).pack(side="left", padx=5)
        ttk.Label(header_frame, text="Room Name", width=20).pack(side="left", padx=5)  # Room Name 컬럼 추가
        ttk.Label(header_frame, text="Min Size", width=15).pack(side="left", padx=5)
        ttk.Label(header_frame, text="Max Size", width=15).pack(side="left", padx=5)

        # 기존 항목들을 가로로 배치
        for key, value in self.constraints.items(section):
            self.add_existing_size_room_entry(key, value)

        # 방 번호와 크기 추가 버튼
        add_button = ttk.Button(frame, text="Add Room", command=self.add_size_room_entry)
        add_button.pack(pady=10)

    def add_existing_size_room_entry(self, room_number, size_range):
        room_number = int(room_number)
        room_name = self.room_names.get(room_number, 'Select')
        min_size, max_size = size_range.split(',')
        self.add_size_room_entry(room_number, room_name, min_size, max_size)

    def add_size_room_entry(self, room_number="", room_name="Select", min_size="", max_size=""):
        row_frame = ttk.Frame(self.size_entries_frame)
        row_frame.pack(fill="x", pady=5)

        room_name_var = tk.StringVar(value=room_name)

        # 방 번호를 read-only Label로 표시
        room_number_label = ttk.Label(row_frame, text=str(room_number) if room_number else "-1", width=5)
        room_number_label.pack(side="left", padx=5)

        # 방 이름을 선택할 수 있는 Combobox
        room_name_combobox = ttk.Combobox(row_frame, textvariable=room_name_var, width=20, state="readonly")
        room_name_combobox['values'] = list(self.room_names.values())
        room_name_combobox.pack(side="left", padx=5)

        # Combobox가 초기화된 후에 값을 설정
        self.size_entries_frame.after(100, lambda: room_name_combobox.set(room_name))

        # 방 이름 선택 시 해당 방 번호를 Label에 업데이트
        room_name_combobox.bind("<<ComboboxSelected>>",
                                lambda e: self.update_room_number_label(room_name_combobox, room_number_label))

        # 최소 크기 입력 필드
        min_size_entry = ttk.Entry(row_frame, width=15)
        min_size_entry.insert(0, min_size)
        min_size_entry.pack(side="left", padx=5)
        self.min_size_entries.append(min_size_entry)

        min_size_label = ttk.Label(row_frame, text="m²")
        min_size_label.pack(side="left", padx=5)

        # 최대 크기 입력 필드
        max_size_entry = ttk.Entry(row_frame, width=15)
        max_size_entry.insert(0, max_size)
        max_size_entry.pack(side="left", padx=5)
        self.max_size_entries.append(max_size_entry)

        max_size_label = ttk.Label(row_frame, text="m²")
        max_size_label.pack(side="left", padx=5)

        # 제거 버튼 추가
        remove_button = ttk.Button(row_frame, text="Remove",
                                   command=lambda: self.remove_size_room_entry(row_frame, room_number_label,
                                                                               min_size_entry, max_size_entry))
        remove_button.pack(side="left", padx=5)

        self.size_room_entries.append(room_number_label)
        self.room_name_entries.append(room_name_combobox)
        self.entry_frames.append(row_frame)

    def remove_size_room_entry(self, row_frame, room_number_label, min_size_entry, max_size_entry):
        # 설정 파일에서 해당 항목 삭제
        section = "SizeRequirements"
        room_number = room_number_label.cget("text")
        if room_number in self.constraints[section]:
            del self.constraints[section][room_number]

        # 행 프레임과 해당 입력 필드들 제거
        row_frame.pack_forget()
        self.size_room_entries.remove(room_number_label)
        self.min_size_entries.remove(min_size_entry)
        self.max_size_entries.remove(max_size_entry)
        self.entry_frames.remove(row_frame)


    def save_size_requirements(self):
        section = "SizeRequirements"
        for i, (room_label, min_size_entry, max_size_entry) in enumerate(
                zip(self.size_room_entries, self.min_size_entries, self.max_size_entries)):
            room_number = room_label.cget("text")  # Label에서 방 번호를 가져옴
            min_size = min_size_entry.get()
            max_size = max_size_entry.get()
            if room_number and min_size and max_size:
                self.constraints[section][room_number] = f"{min_size},{max_size}"

    ############################################################
    # info: adjacency_requirements_section
    ############################################################
    def add_adjacency_requirements_section(self):
        # AdjacencyRequirements 섹션 생성
        section = "AdjacencyRequirements"
        if not self.constraints.has_section(section):
            self.constraints.add_section(section)

        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text='Adjacency')

        self.sections[section] = {}

        self.adjacency_entries_frame = ttk.Frame(frame)
        self.adjacency_entries_frame.pack(fill="x", padx=10, pady=5)

        # 컬럼 제목 추가
        header_frame = ttk.Frame(self.adjacency_entries_frame)
        header_frame.pack(fill="x", pady=5)

        ttk.Label(header_frame, text="", width=2).pack(side="left", padx=5)
        ttk.Label(header_frame, text="Room 1", width=15).pack(side="left", padx=5)  # Room 1 Name 컬럼 추가
        ttk.Label(header_frame, text="", width=2).pack(side="left", padx=20)
        ttk.Label(header_frame, text="Room 2", width=15).pack(side="left", padx=5)  # Room 2 Name 컬럼 추가

        # 기존 항목들을 가로로 배치
        if self.constraints.has_option(section, "adjacent"):
            adjacency_list = ast.literal_eval(self.constraints.get(section, "adjacent"))
            for room1, room2 in adjacency_list:
                self.add_existing_adjacency_entry(room1, room2)

        # 방 번호와 인접 관계 추가 버튼
        add_button = ttk.Button(frame, text="Add Adjacency", command=self.add_adjacency_entry)
        add_button.pack(pady=10)

    def add_existing_adjacency_entry(self, room1, room2):
        room1 = int(room1)  # 방 번호를 정수로 변환
        room2 = int(room2)  # 방 번호를 정수로 변환
        room1_name = self.room_names.get(room1, 'Select')  # 방 이름 가져오기
        room2_name = self.room_names.get(room2, 'Select')  # 방 이름 가져오기
        self.add_adjacency_entry(room1, room2, room1_name, room2_name)

    def add_adjacency_entry(self, room1="", room2="", room1_name="Select", room2_name="Select"):
        row_frame = ttk.Frame(self.adjacency_entries_frame)
        row_frame.pack(fill="x", pady=5)

        room1_name_var = tk.StringVar(value=room1_name)
        room2_name_var = tk.StringVar(value=room2_name)

        # Room 1 번호를 Label로 표시
        room1_label = ttk.Label(row_frame, text=str(room1) if room1 else "0", width=2)
        room1_label.pack(side="left", padx=5)

        # Room 1 이름을 선택할 수 있는 Combobox
        room1_combobox = ttk.Combobox(row_frame, textvariable=room1_name_var, width=15, state="readonly")
        room1_combobox['values'] = list(self.room_names.values())
        room1_combobox.pack(side="left", padx=5)

        # Room 2 번호를 Label로 표시
#        room2_label = ttk.Label(row_frame, text=str(room2) if room2 else "0", width=2)
#        room2_label.pack(side="left", padx=50)

        # info 두 룸 사이의 간격을 넓히고 싶어서 위의 내용을 다음과 같이 변경
        room2_label = ttk.Label(row_frame, text=str(room2) if room2 else "0", width=2, anchor="w")
        room2_label.pack(side="left", padx=(30, 0))  # 왼쪽에만 50의 간격을 주고 오른쪽은 0으로 설정

        # Room 2 이름을 선택할 수 있는 Combobox
        room2_combobox = ttk.Combobox(row_frame, textvariable=room2_name_var, width=15, state="readonly")
        room2_combobox['values'] = list(self.room_names.values())
        room2_combobox.pack(side="left", padx=5)

        # Combobox 선택 시 방 번호를 Label에 업데이트
        room1_combobox.bind("<<ComboboxSelected>>",
                            lambda e: self.update_room_number_label(room1_combobox, room1_label))
        room2_combobox.bind("<<ComboboxSelected>>",
                            lambda e: self.update_room_number_label(room2_combobox, room2_label))

        # 초기 값 설정
        self.adjacency_entries_frame.after(100, lambda: room1_combobox.set(room1_name))
        self.adjacency_entries_frame.after(100, lambda: room2_combobox.set(room2_name))

        self.adjacency_room1_entries.append(room1_label)
        self.adjacency_room2_entries.append(room2_label)

        # 제거 버튼 추가
        remove_button = ttk.Button(row_frame, text="Remove",
                                   command=lambda: self.remove_adjacency_entry(row_frame, room1_label, room2_label))
        remove_button.pack(side="left", padx=5)

        self.entry_frames.append(row_frame)

    def remove_adjacency_entry(self, row_frame, room1_label, room2_label):
        # 설정 파일에서 해당 항목 삭제
        section = "AdjacencyRequirements"

        # 방 번호를 가져오기
        room1 = room1_label.cget("text")
        room2 = room2_label.cget("text")

        # 인접성 리스트를 가져와서 방 번호를 삭제
        adjacency_list = ast.literal_eval(self.constraints.get(section, "adjacent"))
        adjacency_pair = (int(room1), int(room2))

        if adjacency_pair in adjacency_list:
            adjacency_list.remove(adjacency_pair)
            self.constraints.set(section, "adjacent", str(adjacency_list))

        # 행 프레임과 해당 입력 필드들 제거
        row_frame.pack_forget()
        self.adjacency_room1_entries.remove(room1_label)
        self.adjacency_room2_entries.remove(room2_label)
        self.entry_frames.remove(row_frame)

    def save_adjacency_requirements(self):
        section = "AdjacencyRequirements"
        adjacency_list = []
        for i, (room1_label, room2_label) in enumerate(zip(self.adjacency_room1_entries, self.adjacency_room2_entries)):
            room1 = room1_label.cget("text")  # Label에서 방 번호를 가져옴
            room2 = room2_label.cget("text")  # Label에서 방 번호를 가져옴
            if room1 and room2:
                adjacency_list.append((int(room1), int(room2)))
        self.constraints.set(section, "adjacent", str(adjacency_list))

    def save_settings(self):
        # RoomNames, Orientation, Size, Adjacency 등 기타 섹션을 저장하는 로직
        self.save_orientation_requirements()
        self.save_size_requirements()
        self.save_adjacency_requirements()
        self.save_room_names()

        # 그 외의 섹션 (RoomNames를 제외한 모든 섹션)
        for section, items in self.sections.items():
            for key, entry in items.items():
                value = entry.get()
                self.config.set(section, key, value)

        write_ini_file(self.config, 'config.ini')
        write_ini_file(self.constraints, 'constraints.ini')
        messagebox.showinfo("Settings", "Settings saved successfully!")



        def update_room_name(self, room_number_label, room_name_var):
            """방 이름이 변경될 때 호출되는 이벤트 핸들러"""
            room_number = room_number_label.cget("text")
            room_name = room_name_var.get().strip()
            section = "RoomNames"
            if room_number and room_name:
                self.config.set(section, room_number, room_name)
            else:
                if room_number in self.config[section]:
                    del self.config[section][room_number]

def main():
    root = tk.Tk()
    root.title("Main Application")

    def open_settings():
        settings_root = tk.Toplevel(root)
        settings_root.title("Settings")
        config = read_ini_file('config.ini')
        constraints = read_ini_file('constraints.ini')
        SettingsApp(settings_root, config, constraints)

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
    pass
    # main()