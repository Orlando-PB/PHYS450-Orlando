import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from plots import show_fits_info
from fits_processor import process_light_images


def run_gui():
    # Runs GUI
    root = tk.Tk()
    root.title("FITS Image Processor")

    # Ensure the window opens in the foreground
    root.focus_force()

    # Variables to hold toggle states
    use_flats_var = tk.BooleanVar(value=True) # Applies Flat Frames During Calibration
    use_darks_var = tk.BooleanVar(value=True) # Applies Dark Frames During Calibration
    use_biases_var = tk.BooleanVar(value=True) # Applies Bias Frames During Calibration
    stack_var = tk.BooleanVar(value=True) # Stacks after Calibration

    def start_processing():
        # Processing Function
        base_folder = base_folder_entry.get() # Gets folder to use.

        # Create a unique output folder name
        output_number = 1
        while os.path.exists(os.path.join(base_folder, f"Output {output_number}")):
            output_number += 1
        output_folder = os.path.join(base_folder, f"Output {output_number}")
        print(f"===\nUsing {output_folder}!\n===")

        try:
            start_time = time.time()  # Record the start time

            # Main processing function
            
            process_light_images(
                base_folder,
                output_folder,
                use_flats_var.get(),
                use_darks_var.get(),
                use_biases_var.get(),
                stack_var.get(),
            )

            end_time = time.time()  # Record the end time
            processing_time = end_time - start_time  # Calculate total processing time

            # Print processing results to the console
            print("=======\nProcessing complete!\n")
            print(f"Output Saved To: {output_folder}")
            print(f"Time Taken: {processing_time:.2f} seconds.")

        except Exception as e: # In case of error
            messagebox.showerror("Error", str(e))

    # Folder selection
    def select_folder():
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            base_folder_entry.delete(0, tk.END)
            base_folder_entry.insert(0, folder_selected)

    # File selection for viewing statistics
    def select_file_and_show_info():
        file_selected = filedialog.askopenfilename(
            filetypes=[("FITS files", "*.fit *.fits")]
        )
        if file_selected:
            try:
                show_fits_info(file_selected)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # GUI Layout
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    tk.Label(frame, text="Base Folder:").grid(row=0, column=0, sticky="e")
    base_folder_entry = tk.Entry(frame, width=50)
    base_folder_entry.insert(0, "./demo small")
    base_folder_entry.grid(row=0, column=1, padx=5)

    tk.Button(frame, text="Browse", command=select_folder).grid(row=0, column=2, padx=5)

    tk.Checkbutton(frame, text="Use Flats", variable=use_flats_var).grid(
        row=1, column=0, sticky="w"
    )
    tk.Checkbutton(frame, text="Use Darks", variable=use_darks_var).grid(
        row=2, column=0, sticky="w"
    )
    tk.Checkbutton(frame, text="Use Biases", variable=use_biases_var).grid(
        row=3, column=0, sticky="w"
    )

    tk.Button(frame, text="Start Processing", command=start_processing).grid(
        row=6, column=1, pady=10
    )
    tk.Button(
        frame,
        text="Open FITS and Show Info",
        command=select_file_and_show_info,
    ).grid(row=7, column=1, pady=10)

    root.mainloop()

run_gui() # Run Code
