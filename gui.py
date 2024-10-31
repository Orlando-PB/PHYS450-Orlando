import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from plots import show_fits_info  # Import the new function
from datetime import datetime

# Initialize base folder globally
base_folder = "./demo"

def run_gui(process_func):
    root = tk.Tk()
    root.title("FITS Image Processor")

    # Ensure the window opens in the foreground
    root.lift()  # Raise the window to the front
    root.focus_force()  # Force focus on the window

    # Variables to hold toggle states
    use_flats_var = tk.BooleanVar(value=True)
    use_darks_var = tk.BooleanVar(value=True)
    use_biases_var = tk.BooleanVar(value=True)
    stack_var = tk.BooleanVar(value=True)

    def start_processing():
        global base_folder  # Ensure we use the updated base_folder

        # Create a unique output folder name using the current date and time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join(base_folder, f"Output_{timestamp}")

        # Ensure the unique output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        try:
            start_time = time.time()  # Record the start time
            
            # Start the main processing function
            total_files = process_func(base_folder, output_folder, use_flats_var.get(), use_darks_var.get(), use_biases_var.get(), stack_var.get())
            
            # If plotting histograms is enabled
            if plot_histograms_var.get():
                flat_folder = os.path.join(output_folder, "calibration")  # Ensure this points to the output's calibration folder
                hist_output_folder = os.path.join(output_folder, "histograms")
                plot_histograms_for_master_flats(flat_folder, hist_output_folder)
            
            end_time = time.time()  # Record the end time
            processing_time = end_time - start_time  # Calculate the total processing time
            
            messagebox.showinfo("Success", f"Processing complete. {total_files} images processed.\n"
                                           f"Output saved to {output_folder}\n"
                                           f"Time taken: {processing_time:.2f} seconds.")
        except Exception as e:
            messagebox.showerror("Error", str(e))


    # Folder selection
    def select_folder():
        global base_folder  # Ensure we can modify the base_folder
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            base_folder_entry.delete(0, tk.END)
            base_folder_entry.insert(0, folder_selected)
            base_folder = folder_selected  # Update base_folder

    # File selection for viewing statistics
    def select_file_and_show_info():
        file_selected = filedialog.askopenfilename(filetypes=[("FITS files", "*.fit *.fits")])
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
    base_folder_entry.insert(0, base_folder)
    base_folder_entry.grid(row=0, column=1, padx=5)

    tk.Button(frame, text="Browse", command=select_folder).grid(row=0, column=2, padx=5)

    tk.Checkbutton(frame, text="Use Flats", variable=use_flats_var).grid(row=1, column=0, sticky="w")
    tk.Checkbutton(frame, text="Use Darks", variable=use_darks_var).grid(row=2, column=0, sticky="w")
    tk.Checkbutton(frame, text="Use Biases", variable=use_biases_var).grid(row=3, column=0, sticky="w")
    tk.Checkbutton(frame, text="Stack", variable=stack_var).grid(row=4, column=0, sticky="w")

    tk.Button(frame, text="Start Processing", command=start_processing).grid(row=6, column=1, pady=10)
    tk.Button(frame, text="Open FITS and Show Info", command=select_file_and_show_info).grid(row=7, column=1, pady=10)  # New button to open FITS and show info

    root.mainloop()
