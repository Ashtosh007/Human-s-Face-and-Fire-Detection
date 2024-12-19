import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox
import subprocess


def capture_faces():
    """
    Function to start the face capturing script.
    """
    try:
        subprocess.run(["python3", "/home/ashutosh007/Desktop/proj/FaceCapturing.py"])  
    except Exception as e:
        messagebox.showerror("Error", f"Error starting face capture: {str(e)}")

def Face_detect():
    """
    Function to start the face detecting script.
    """
    try:
        subprocess.run(["python3", "/home/ashutosh007/Desktop/proj/Face_Recognise.py"])  
    except Exception as e:
        messagebox.showerror("Error", f"Error starting face detecting: {str(e)}")


def run_detection_tool():
    """
    Function to start the detection tool script.
    """
    try:
        subprocess.run(["python3", "/home/ashutosh007/Desktop/proj/Human_Fire_Detection.py"])  
    except Exception as e:
        messagebox.showerror("Error", f"Error starting detection tool: {str(e)}")


# Create the main GUI window
root = tk.Tk()
root.title("Face & Fire Detection Tool")
root.geometry("500x400")
# root.config(bg="#2e3f4f") -- removing or not using  becoz i had used background imge. 
root.resizable(False, False) 

# Bg_Imge:-
bg_image = ImageTk.PhotoImage(Image.open("/home/ashutosh007/Desktop/proj/8.jpg").resize((500, 400)))
tk.Label(root, image=bg_image).place(relwidth=1, relheight=1)



title_label = tk.Label(root, text="Face & Fire Detection Software", font=("Times New Roman", 18), bg="#2e3f4f", fg="white")
title_label.pack(pady=20)

# Button to start face capturing
capture_button = tk.Button(
    root,
    text="Face Capturing",
    font=("Times New Roman", 12, "bold"),
    bg="#7171C6",
    fg="white",
    command=capture_faces
)
capture_button.pack(pady=20)

#button for detect face:- 
face_detection_button = tk.Button(
    root,
    text="Face Detecting",
    font=("Times New Roman", 12, "bold"),
    bg="#4CAF50",
    fg="white",
    command=Face_detect
)
face_detection_button.pack(pady=20)

# Button to start detection tool
detection_button = tk.Button(
    root,
    text="Face & fire Detection Tool",
    font=("Times New Roman", 12, "bold"),
    bg="#f44336",
    fg="white",
    command=run_detection_tool
)
detection_button.pack(pady=20)


quit_button = tk.Button(
    root,
    text="Quit",
    font=("Times New Roman", 12, "bold"),
    bg="#555555",
    fg="white",
    command=root.quit
)
quit_button.pack(pady=20)


root.mainloop()
