import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
from config_reader import read_config_int, read_str_value_of

class GridGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Grid Generator")

        # 기본 설정
        self.pixel_size = 20  # todo: settings.ini에서 설정값 가져오기
        self.rows, self.cols = read_config_int('config.ini', 'Metrics', 'base_rows'), read_config_int('config.ini', 'Metrics', 'base_cols')
        self.default_color = "white"
        self.selected_color = "gray"
        self.highlight_color = "lightgray"
        self.matrix = self.initialize_matrix(self.rows, self.cols)

        # 하이라이트된 셀 추적을 위한 집합
        self.highlighted_cells = set()

        # 캔버스 생성
        self.canvas = tk.Canvas(root)
        self.canvas.pack()
        self.create_grid_structure()  # 초기 그리드 생성

        # 메뉴 생성
        self.create_menu()

        # 드래그 시작/종료 위치 초기화
        self.start_pos = None

        # 이벤트 바인딩
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<B1-Motion>", self.on_drag)

    def initialize_matrix(self, rows, cols):
        """0으로 초기화된 행렬을 반환합니다."""
        return [[0 for _ in range(cols)] for _ in range(rows)]

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File 메뉴 생성
        filemenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='File', menu=filemenu)

        # New 메뉴 항목 추가
        filemenu.add_command(label='New', command=self.new_grid_dialog)
        # Save 메뉴 항목 추가
        filemenu.add_command(label='Save', command=self.save_grid)
        #load 메뉴 항목 추가
        filemenu.add_command(label='Load', command=self.load_grid)

    def new_grid_dialog(self):
        def on_confirm():
            try:
                rows = int(rows_entry.get())
                cols = int(cols_entry.get())
                scale = float(scale_entry.get())
                pixel_size = int(pixel_size_entry.get())
                if rows > 0 and cols > 0 and pixel_size > 0 and scale > 0 :
                    self.rows, self.cols = rows, cols
                    self.scale = scale  # 스케일 업데이트
                    self.pixel_size = pixel_size  # 픽셀 크기 업데이트

                    self.matrix = self.initialize_matrix(rows, cols)
                    self.create_grid_structure()  # 새 셀 구조 생성
                    self.update_grid_display()  # 셀 표시 업데이트

                    dialog.destroy()
                else:
                    messagebox.showerror('Invalid Input', 'All values must be positive numbers.')
            except ValueError:
                messagebox.showerror('Invalid Input', 'Please enter valid integers or floats.')


        dialog = tk.Toplevel(self.root)
        dialog.title('New Grid Size')
        dialog.geometry('350x250')
        dialog.grab_set()  # 모달 창 설정

        # Rows와 Columns를 나란히 배치할 Frame 생성
        row_col_frame = tk.Frame(dialog)
        row_col_frame.pack(pady=(20, 10))

        # Rows 입력 필드
        tk.Label(row_col_frame, text="Rows and columns:").grid(row=0, column=0, padx=(0, 5))
        rows_entry = tk.Entry(row_col_frame, width=4)
        rows_entry.insert(0, str(self.rows))
        rows_entry.grid(row=0, column=1)

        # Columns 입력 필드
        tk.Label(row_col_frame, text="X ").grid(row=0, column=2, padx=(10, 5))
        cols_entry = tk.Entry(row_col_frame, width=5)
        cols_entry.insert(0, str(self.cols))
        cols_entry.grid(row=0, column=3)

        # Cell Length (Scale)과 Pixel Size를 위한 Frame 생성
        size_frame = tk.Frame(dialog)
        size_frame.pack(pady=10)

        # Cell Length (Scale) 입력 필드
        tk.Label(size_frame, text="Scale: Cell Length").grid(row=0, column=0, padx=(0, 5))
        scale_entry = tk.Entry(size_frame, width=10)
        scale_entry.insert(0, str(read_config_int('config.ini', 'Metrics', 'scale')))
        # cell_size_entry.insert(0, str(self.cell_size))  # 현재 cell_size 값 사용
        scale_entry.grid(row=0, column=1)

        # mm² 단위 표시
        tk.Label(size_frame, text="mm²").grid(row=0, column=3, padx=(5, 0))

        # Pixel Size 입력 필드
        tk.Label(size_frame, text="Screen Cell Size:").grid(row=1, column=0, padx=(0, 5), pady=(10, 0))
        pixel_size_entry = tk.Entry(size_frame, width=10)
        pixel_size_entry.insert(0, str(read_config_int('config.ini', 'Metrics', 'unit_pixel_size')))
        pixel_size_entry.grid(row=1, column=1, pady=(10, 0))
        tk.Label(size_frame, text="pixel").grid(row=1, column=3, padx=(5, 0))

        # Default Grid File 입력 필드
        tk.Label(size_frame, text='Default Footprint Grid:').grid(row=2, column=0, padx=(0, 5), pady=(10,0))
        grid_file_entry = tk.Entry(size_frame, width = 10)
        grid_file_entry.insert(0, str(read_str_value_of('config.ini', 'FileSettings', 'default_grid_file')))
        grid_file_entry.grid(row=2, column=1, pady=(10, 0))

        # Confirm 버튼
        tk.Button(dialog, text='Confirm', command=on_confirm).pack(pady=30)
        dialog.transient(self.root)

    def create_grid_structure(self):
        """Canvas에 행렬 크기에 맞게 셀을 생성합니다."""
        self.canvas.delete("all")  # 기존 그리드 제거
        width, height = self.cols * self.pixel_size, self.rows * self.pixel_size
        self.canvas.config(width=width, height=height)

        # 셀 생성
        self.cells = {}
        for row in range(self.rows):
            for col in range(self.cols):
                x1 = col * self.pixel_size
                y1 = row * self.pixel_size
                x2 = x1 + self.pixel_size
                y2 = y1 + self.pixel_size
                cell_id = self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.default_color)
                self.cells[(row, col)] = cell_id


    def update_grid_display(self):
        for row in range(self.rows):
            for col in range(self.cols):
                cell_id = self.cells.get((row, col))
                if cell_id:
                    color = self.selected_color if self.matrix[row][col] == 1 else self.default_color
                    self.canvas.itemconfig(cell_id, fill=color)

    def on_press(self, event):
        # 드래그 시작 위치 설정 및 하이라이트 초기화
        self.start_pos = (event.y // self.pixel_size, event.x // self.pixel_size)
        self.clear_highlight()
        self.highlighted_cells.clear()  # 드래그 시작 시 하이라이트된 셀 초기화

    def on_drag(self, event):
        # 드래그 중 마우스가 지나가는 셀들을 하이라이트
        end_row, end_col = event.y // self.pixel_size, event.x // self.pixel_size
        self.highlight_cells(self.start_pos, (end_row, end_col))

    def on_release(self, event):
        # 드래그가 끝난 후 하이라이트된 셀들만 선택된 상태로 변경
        for row, col in self.highlighted_cells:
            if 0 <= row < self.rows and 0 <= col < self.cols:
                self.matrix[row][col] = 1  # 선택된 셀의 값을 1로 설정
                cell_id = self.cells.get((row, col))
                if cell_id:
                    self.canvas.itemconfig(cell_id, fill=self.selected_color)  # 색상 업데이트

        # 초기화
        self.highlighted_cells.clear()
        self.start_pos = None

    def highlight_cells(self, start_pos, end_pos):
        """주어진 두 위치 사이의 셀들을 하이라이트하고 추적합니다."""
        start_row, start_col = start_pos
        end_row, end_col = end_pos

        # 드래그 범위 내 모든 셀을 하이라이트
        for row in range(min(start_row, end_row), max(start_row, end_row) + 1):
            for col in range(min(start_col, end_col), max(start_col, end_col) + 1):
                if 0 <= row < self.rows and 0 <= col < self.cols:
                    cell_id = self.cells[(row, col)]
                    self.canvas.itemconfig(cell_id, fill=self.highlight_color)
                    self.highlighted_cells.add((row, col))  # 하이라이트된 셀 좌표 추가

    def clear_highlight(self):
        """하이라이트된 셀을 기본 색으로 초기화합니다."""
        for (row, col), cell_id in self.cells.items():
            color = self.selected_color if self.matrix[row][col] == 1 else self.default_color
            self.canvas.itemconfig(cell_id, fill=color)

    def toggle_cell_by_position(self, row, col):
        # 행렬 값 토글
        self.matrix[row][col] = 1 - self.matrix[row][col]

        # 셀 색상 업데이트
        color = self.selected_color if self.matrix[row][col] == 1 else self.default_color
        cell_id = self.cells[(row, col)]
        self.canvas.itemconfig(cell_id, fill=color)

    def get_matrix(self):
        return self.matrix

    def save_grid(self):
        file_path = filedialog.asksaveasfilename(defaultextension='.grd', filetypes=[("Grid files", "*.grd"), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'w') as file:
                    for row in self.matrix:
                        file.write(' '.join(map(str, row)) + '\n')
                messagebox.showinfo('Save', 'Grid successfully saved')
            except Exception as e:
                messagebox.showerror('Save', f"Error writing file to {file_path}: {e}")


    def load_grid(self):
        file_path = filedialog.askopenfilename(defaultextension='.grd',
                                               filetypes=[("Grid files", "*.grd"), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    matrix = [list(map(int, line.split())) for line in lines]

                    # 행렬 크기를 새로 불러온 데이터 크기에 맞게 조정
                    self.rows, self.cols = len(matrix), len(matrix[0]) if matrix else 0
                    self.matrix = matrix
                    self.create_grid_structure()  # 새로운 행렬로 그리드 생성
                    self.update_grid_display()
            except Exception as e:
                messagebox.showerror('Load', f"Error reading file from {file_path}: {e}")

    import numpy as np
    @staticmethod
    def load_grid_as_np(file_path):
        """
        주어진 .grd 파일을 불러와 numpy 배열로 반환합니다.

        Args:
            file_path (str): .grd 파일의 경로

        Returns:
            np.array: 파일의 2D 배열 데이터를 담은 numpy 배열
        """
        try:
            with open(file_path, 'r') as file:
                # 파일을 읽고 각 줄을 정수 리스트로 변환하여 2D 리스트로 저장
                matrix = [list(map(int, line.split())) for line in file.readlines()]
            # numpy 배열로 변환하여 반환
            return np.array(matrix)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None


def run_grid_generator():
    root = tk.Tk()
    app = GridGenerator(root)
    print(app.get_matrix())
    root.mainloop()

# if __name__ == '__main__':
#   run_grid_generator()
#