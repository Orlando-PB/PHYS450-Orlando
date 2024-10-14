import tkinter as tk
from tkinter import ttk, filedialog, messagebox

def run_gui(base_folder, output_folder, process_func):
    root = tk.Tk()
    root.title("FITS Image Processor")

    # Variables to hold toggle states
    use_flats_var = tk.BooleanVar(value=True)  # Default: use flats
    use_darks_var = tk.BooleanVar(value=True)  # Default: use darks
    use_biases_var = tk.BooleanVar(value=True)  # Default: use biases
    combine_lrgb_var = tk.BooleanVar(value=False)  # Default: don't combine LRGB

    # Create a progress bar
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate", variable=progress_var)

    def start_processing():
        try:
            total_files = process_func(base_folder, output_folder, use_flats_var.get(), use_darks_var.get(), use_biases_var.get(), combine_lrgb_var.get(), progress_var, progress_bar)
            progress_var.set(100)  # Set the progress bar to 100% when done
            messagebox.showinfo("Success", f"Processing complete. {total_files} images processed. Output saved to {output_folder}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Folder selection (optional, if base_folder should be user-configurable)
    def select_folder():
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            base_folder_entry.delete(0, tk.END)
            base_folder_entry.insert(0, folder_selected)

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
    tk.Checkbutton(frame, text="Combine LRGB", variable=combine_lrgb_var).grid(row=4, column=0, sticky="w")

    tk.Button(frame, text="Start Processing", command=start_processing).grid(row=5, column=1, pady=10)

    progress_bar.pack(pady=10)  # Add the progress bar to the GUI

    root.mainloop()
