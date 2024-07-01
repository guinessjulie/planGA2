import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import PyPDF2
import os

def convert_jpg_to_pdf(folder_path):
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                jpg_file = os.path.join(folder_path, filename)
                pdf_file = os.path.join(folder_path, filename.replace('.jpg', '.pdf').replace('.jpeg', '.pdf'))
                image = Image.open(jpg_file)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(pdf_file, 'PDF', resolution=100.0)
        messagebox.showinfo("Success", "All JPG files have been converted to PDFs.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def merge_pdfs(paths, output):
    try:
        pdf_writer = PyPDF2.PdfWriter()
        for path in paths:
            pdf_reader = PyPDF2.PdfReader(path)
            for page in range(len(pdf_reader.pages)):
                pdf_writer.add_page(pdf_reader.pages[page])
        with open(output, 'wb') as out:
            pdf_writer.write(out)
        messagebox.showinfo("Success", f"Merged PDF saved as: {output}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def select_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        convert_jpg_to_pdf(folder_path)

def select_pdfs():
    file_paths = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
    if file_paths:
        output_file = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
        if output_file:
            merge_pdfs(file_paths, output_file)
def gui_main():
    app = tk.Tk()
    app.title("JPG to PDF & Merge PDFs")

    convert_button = tk.Button(app, text="Convert JPG to PDF", command=select_folder)
    convert_button.pack(pady=10)

    merge_button = tk.Button(app, text="Merge PDF Files", command=select_pdfs)
    merge_button.pack(pady=10)

    app.mainloop()

