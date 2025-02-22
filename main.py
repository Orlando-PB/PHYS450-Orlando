import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont

from fits_processor import process_light_images
from plots import FITSViewer

# Dark theme colors
BG_COLOR = "#2B2B2B"  # main background
FG_COLOR = "#FFFFFF"  # text color

def run_app():
    root = tk.Tk()
    root.title("Simple Astronomy")
    root.geometry("1200x800")
    root.configure(bg=BG_COLOR)
    
    # Setup ttk style for dark theme
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
    style.configure("TNotebook.Tab", background=BG_COLOR, foreground=FG_COLOR)
    style.map("TNotebook.Tab", background=[("selected", "#3A3A3A")])
    style.configure("TFrame", background=BG_COLOR)
    style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR)
    style.configure("TButton", background=BG_COLOR, foreground=FG_COLOR)
    style.map("TButton", background=[("active", "#444444")])
    style.configure("TCheckbutton", background=BG_COLOR, foreground=FG_COLOR)
    
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both", padx=0, pady=0)
    
    # ---------------- Process Tab ----------------
    process_tab = ttk.Frame(notebook)
    notebook.add(process_tab, text="Process")
    
    use_flats_var = tk.BooleanVar(value=True)
    use_darks_var = tk.BooleanVar(value=True)
    use_biases_var = tk.BooleanVar(value=True)
    stack_var = tk.BooleanVar(value=True)
    
    def start_processing():
        base_folder = base_folder_entry.get()
        output_number = 1
        while os.path.exists(os.path.join(base_folder, f"Output {output_number}")):
            output_number += 1
        output_folder = os.path.join(base_folder, f"Output {output_number}")
        print(f"Using {output_folder}!")
        try:
            import time
            start_time = time.time()
            process_light_images(
                base_folder,
                output_folder,
                use_flats_var.get(),
                use_darks_var.get(),
                use_biases_var.get(),
                stack_var.get(),
            )
            end_time = time.time()
            print("Processing complete!")
            print(f"Time Taken: {end_time - start_time:.2f} seconds.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def select_folder():
        folder = filedialog.askdirectory()
        if folder:
            base_folder_entry.delete(0, tk.END)
            base_folder_entry.insert(0, folder)
    
    proc_frame = ttk.Frame(process_tab)
    proc_frame.pack(fill="both", expand=True, padx=10, pady=10)
    ttk.Label(proc_frame, text="Base Folder:").grid(row=0, column=0, sticky="e")
    base_folder_entry = ttk.Entry(proc_frame, width=50)
    base_folder_entry.insert(0, "./demo small")
    base_folder_entry.grid(row=0, column=1, padx=5)
    ttk.Button(proc_frame, text="Browse", command=select_folder).grid(row=0, column=2, padx=5)
    ttk.Checkbutton(proc_frame, text="Use Flats", variable=use_flats_var).grid(row=1, column=0, sticky="w")
    ttk.Checkbutton(proc_frame, text="Use Darks", variable=use_darks_var).grid(row=2, column=0, sticky="w")
    ttk.Checkbutton(proc_frame, text="Use Biases", variable=use_biases_var).grid(row=3, column=0, sticky="w")
    ttk.Button(proc_frame, text="Start Processing", command=start_processing).grid(row=4, column=1, pady=10)
    
    # ---------------- Viewer Tab ----------------
    viewer_tab = ttk.Frame(notebook)
    notebook.add(viewer_tab, text="Viewer")
    
    # Top controls: Open FITS button and filename display side by side.
    controls_frame = ttk.Frame(viewer_tab)
    controls_frame.pack(side="top", fill="x", padx=5, pady=5)
    
    # Create a font for the filename label
    default_font = tkfont.nametofont("TkDefaultFont")
    filename_font = tkfont.Font(root, default_font)
    filename_font.configure(overstrike=0)
    
    viewer_instance = None
    
    def close_fit_file(event):
        nonlocal viewer_instance
        if viewer_instance is not None:
            for child in plot_frame.winfo_children():
                child.destroy()
            viewer_instance = None
            filename_label.config(text="No file selected")
    
    def on_filename_enter(event):
        filename_label.config(foreground="red")
        filename_font.configure(overstrike=1)
        filename_label.config(font=filename_font)
    
    def on_filename_leave(event):
        filename_label.config(foreground=FG_COLOR)
        filename_font.configure(overstrike=0)
        filename_label.config(font=filename_font)
    
    def open_fits():
        nonlocal viewer_instance
        file_selected = filedialog.askopenfilename(filetypes=[("FITS files", "*.fit *.fits")])
        if file_selected:
            filename_label.config(text=os.path.basename(file_selected))
            # Clear previous canvas if any
            for child in plot_frame.winfo_children():
                child.destroy()
            viewer_instance = FITSViewer(file_selected, plot_frame, stats_labels)
    
    open_btn = ttk.Button(controls_frame, text="Open FITS File", command=open_fits)
    open_btn.pack(side="left", padx=10)
    filename_label = ttk.Label(controls_frame, text="No file selected", width=30)
    filename_label.pack(side="left", padx=10)
    filename_label.bind("<Enter>", on_filename_enter)
    filename_label.bind("<Leave>", on_filename_leave)
    filename_label.bind("<Button-1>", close_fit_file)
    
    # Main viewer area: split into left (2/3) and right (1/3)
    viewer_content = ttk.Frame(viewer_tab)
    viewer_content.pack(fill="both", expand=True)
    viewer_content.columnconfigure(0, weight=2)
    viewer_content.columnconfigure(1, weight=1, minsize=300)
    viewer_content.rowconfigure(0, weight=1)
    
    # Left frame for the Matplotlib canvas (image + histogram)
    plot_frame = ttk.Frame(viewer_content)
    plot_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    
    # Right frame for stats and buttons (fixed layout)
    stats_frame = ttk.Frame(viewer_content)
    stats_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
    
    # Fixed-size labels for stats
    image_stats_label = ttk.Label(stats_frame, text="Image Stats:\n", anchor="nw", justify="left", width=40)
    image_stats_label.pack(padx=5, pady=5)
    cursor_stats_label = ttk.Label(stats_frame, text="Cursor Stats:\n", anchor="nw", justify="left", width=40)
    cursor_stats_label.pack(padx=5, pady=5)
    selected_stats_label = ttk.Label(stats_frame, text="Selected Source Stats:\nNo source selected.\nClick a source to see details.", anchor="nw", justify="left", width=40)
    selected_stats_label.pack(padx=5, pady=5)
    
    # Container frame for interactive buttons (pinned at bottom)
    btns_frame = ttk.Frame(stats_frame)
    btns_frame.pack(side="bottom", fill="x", padx=5, pady=5)
    
    # Zoom controls: three icon buttons arranged horizontally
    zoom_frame = ttk.Frame(btns_frame)
    zoom_frame.pack(fill="x", pady=2)
    ttk.Button(zoom_frame, text="🔍➕", command=lambda: viewer_instance and viewer_instance.zoom_in()).pack(side="left", expand=True, fill="x", padx=2)
    ttk.Button(zoom_frame, text="🔍➖", command=lambda: viewer_instance and viewer_instance.zoom_out()).pack(side="left", expand=True, fill="x", padx=2)
    ttk.Button(zoom_frame, text="↺", command=lambda: viewer_instance and viewer_instance.reset_zoom()).pack(side="left", expand=True, fill="x", padx=2)
    
    # Other interactive buttons
    ttk.Button(btns_frame, text="Toggle Auto Stretch", command=lambda: viewer_instance and viewer_instance.toggle_auto_stretch()).pack(fill="x", pady=2)
    ttk.Button(btns_frame, text="Toggle Log Hist", command=lambda: viewer_instance and viewer_instance.toggle_log()).pack(fill="x", pady=2)
    ttk.Button(btns_frame, text="Toggle Sources", command=lambda: viewer_instance and viewer_instance.toggle_sources()).pack(fill="x", pady=2)
    
    # Dictionary of stats labels to pass to FITSViewer
    stats_labels = {
        "image": image_stats_label,
        "cursor": cursor_stats_label,
        "selected": selected_stats_label
    }
    
    root.mainloop()

if __name__ == "__main__":
    run_app()
