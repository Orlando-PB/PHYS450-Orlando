import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont
from PIL import ImageTk, Image, ImageOps # Pillow for loading icons

from fits_processor import process_light_images
from plots import FITSViewer

# Dark theme colors
BG_COLOR = "#2B2B2B"
FG_COLOR = "#FFFFFF"

def load_icon(path, size=(20,20)):
    """Load a PNG icon from file, convert a black icon to white, and return a PhotoImage."""
    try:
        img = Image.open(path)
        try:
            resample_method = Image.Resampling.LANCZOS
        except AttributeError:
            resample_method = Image.LANCZOS  # For older Pillow versions
        img = img.resize(size, resample_method)
        # If the image has an alpha channel, process RGB channels separately.
        if img.mode == "RGBA":
            r, g, b, a = img.split()
            rgb_img = Image.merge("RGB", (r, g, b))
            inverted = ImageOps.invert(rgb_img)
            img = Image.merge("RGBA", (*inverted.split(), a))
        elif img.mode == "RGB":
            img = ImageOps.invert(img)
        return ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"Error loading icon {path}: {e}")
        return None

def run_app():
    root = tk.Tk()
    root.title("Simple Astronomy")
    root.geometry("1200x800")
    root.configure(bg=BG_COLOR)
    
    # Setup ttk style; force uniform tab sizes
    style = ttk.Style(root)
    style.theme_use("clam")

    # All tabs share the same size/spacing whether selected or not:
    style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        background=BG_COLOR,
        foreground=FG_COLOR,
        uniform="tab",
        focuscolor=BG_COLOR,
        borderwidth=0,
    )
    style.map("TNotebook.Tab", background=[("selected", "#3A3A3A")])
    
    style.configure("TFrame", background=BG_COLOR)
    style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR)
    style.configure("TButton", background=BG_COLOR, foreground=FG_COLOR)
    style.map("TButton", background=[("active", "#444444")])
    style.configure("TCheckbutton", background=BG_COLOR, foreground=FG_COLOR)
    style.map("TCheckbutton", background=[("active", BG_COLOR)])
    
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both", padx=0, pady=0)
    
    # --------------- PROCESS TAB ---------------
    process_tab = ttk.Frame(notebook)
    notebook.add(process_tab, text="Process")
    
    proc_frame = ttk.Frame(process_tab)
    proc_frame.pack(expand=True)
    proc_frame.place(relx=0.5, rely=0.5, anchor="center")
    
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

    # Example icons for the Process tab
    # (Replace these with your chosen icons, e.g. folder-open, arrow-right, etc.)
    icon_browse = load_icon("./icons/folder-open-solid.png", (20,20))
    icon_start  = load_icon("./icons/arrow-right-solid.png", (20,20))

    ttk.Label(proc_frame, text="Base Folder:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    base_folder_entry = ttk.Entry(proc_frame, width=50)
    base_folder_entry.insert(0, "./demo small")
    base_folder_entry.grid(row=0, column=1, padx=5, pady=5)
    
    # "Browse" button w/ an icon
    ttk.Button(
        proc_frame,
        image=icon_browse,
        compound="left",
        text="Browse",
        command=select_folder
    ).grid(row=0, column=2, padx=5, pady=5)
    
    ttk.Checkbutton(proc_frame, text="Use Flats", variable=use_flats_var).grid(row=1, column=0, sticky="w", padx=5, pady=5)
    ttk.Checkbutton(proc_frame, text="Use Darks", variable=use_darks_var).grid(row=2, column=0, sticky="w", padx=5, pady=5)
    ttk.Checkbutton(proc_frame, text="Use Biases", variable=use_biases_var).grid(row=3, column=0, sticky="w", padx=5, pady=5)

    # "Start Processing" button w/ an icon
    ttk.Button(
        proc_frame,
        image=icon_start,
        compound="left",
        text="Start Processing",
        command=start_processing
    ).grid(row=4, column=1, pady=10)
    
    # --------------- VIEWER TAB ---------------
    viewer_tab = ttk.Frame(notebook)
    notebook.add(viewer_tab, text="Viewer")
    
    # Top row: "Open FITS File" + filename label
    controls_frame = ttk.Frame(viewer_tab)
    controls_frame.pack(side="top", fill="x", padx=5, pady=5)
    
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
            # Reset stats
            image_value_label.config(text="Total Pixels: -\nMean: -\nMedian: -\nStd Dev: -\nMin: -\nMax: -")
            cursor_value_label.config(text="RA: -\nDEC: -\nPixel Value: -")
            selected_value_label.config(text="‑")
    
    def on_filename_enter(event):
        if filename_label.cget("text") != "No file selected":
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
            for child in plot_frame.winfo_children():
                child.destroy()
            viewer_instance = FITSViewer(file_selected, plot_frame, stats_labels)

    # Example icons for the Viewer tab
    icon_open_fits = load_icon("./icons/folder-open-solid.png", (20,20))

    open_btn = ttk.Button(
        controls_frame,
        image=icon_open_fits,
        compound="left",
        text="Open FITS File",
        command=open_fits
    )
    open_btn.pack(side="left", padx=10)

    filename_label = ttk.Label(controls_frame, text="No file selected", width=50)
    filename_label.pack(side="left", padx=10)
    filename_label.bind("<Enter>", on_filename_enter)
    filename_label.bind("<Leave>", on_filename_leave)
    filename_label.bind("<Button-1>", close_fit_file)
    
    # Main viewer area
    viewer_content = ttk.Frame(viewer_tab)
    viewer_content.pack(fill="both", expand=True)
    viewer_content.columnconfigure(0, weight=2)
    viewer_content.columnconfigure(1, weight=1, minsize=300)
    viewer_content.rowconfigure(0, weight=1)
    
    # Left side: 800x600 for the Matplotlib figure
    plot_frame = ttk.Frame(viewer_content, width=800, height=600)
    plot_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    
    # Right side: stats + interactive buttons
    stats_frame = ttk.Frame(viewer_content)
    stats_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
    
    # Stats area
    image_header_label = ttk.Label(stats_frame, text="Image Stats:", anchor="nw", justify="left", font=("TkDefaultFont", 10, "bold"))
    image_header_label.pack(padx=5, pady=(5,0), anchor="w")
    image_value_label = ttk.Label(stats_frame, text="Total Pixels: -\nMean: -\nMedian: -\nStd Dev: -\nMin: -\nMax: -", anchor="nw", justify="left", foreground="#CCCCCC", width=40)
    image_value_label.pack(padx=5, pady=(0,5), anchor="w")

    cursor_header_label = ttk.Label(stats_frame, text="Cursor Stats:", anchor="nw", justify="left", font=("TkDefaultFont", 10, "bold"))
    cursor_header_label.pack(padx=5, pady=(5,0), anchor="w")
    cursor_value_label = ttk.Label(stats_frame, text="RA: -\nDEC: -\nPixel Value: -", anchor="nw", justify="left", foreground="#CCCCCC", width=40)
    cursor_value_label.pack(padx=5, pady=(0,5), anchor="w")
    
    selected_header_label = ttk.Label(stats_frame, text="Selected Source Stats:", anchor="nw", justify="left", font=("TkDefaultFont", 10, "bold"))
    selected_header_label.pack(padx=5, pady=(5,0), anchor="w")
    selected_value_label = ttk.Label(stats_frame, text="‑", anchor="nw", justify="left", foreground="#CCCCCC", width=40)
    selected_value_label.pack(padx=5, pady=(0,5), anchor="w")
    
    stats_labels = {
        "image": image_value_label,
        "cursor": cursor_value_label,
        "selected": selected_value_label
    }
    
    # Bottom row: Zoom buttons + toggles
    btns_frame = ttk.Frame(stats_frame)
    btns_frame.pack(side="bottom", fill="x", padx=5, pady=5)
    
    # Load icons for zoom buttons
    icon_zoom_in  = load_icon("./icons/magnifying-glass-plus-solid.png", (20,20))
    icon_zoom_out = load_icon("./icons/magnifying-glass-minus-solid.png", (20,20))
    icon_reset    = load_icon("./icons/rotate-left-solid.png", (20,20))

    # Example icons for toggles:
    # e.g., a "sliders" icon for auto-stretch, a "bars" icon for log hist, etc.
    # Modify as you see fit:
    icon_toggle_stretch = load_icon("./icons/sliders-solid.png", (20,20))
    icon_toggle_log      = load_icon("./icons/bars-solid.png", (20,20))
    icon_toggle_sources  = load_icon("./icons/circle-solid.png", (20,20))
    
    zoom_frame = ttk.Frame(btns_frame)
    zoom_frame.pack(fill="x", pady=2)
    
    ttk.Button(
        zoom_frame,
        image=icon_zoom_in,
        compound="left",
        text="Zoom In",
        command=lambda: viewer_instance and viewer_instance.zoom_in()
    ).pack(side="left", expand=True, fill="x", padx=2)
    
    ttk.Button(
        zoom_frame,
        image=icon_zoom_out,
        compound="left",
        text="Zoom Out",
        command=lambda: viewer_instance and viewer_instance.zoom_out()
    ).pack(side="left", expand=True, fill="x", padx=2)
    
    ttk.Button(
        zoom_frame,
        image=icon_reset,
        compound="left",
        text="Reset",
        command=lambda: viewer_instance and viewer_instance.reset_zoom()
    ).pack(side="left", expand=True, fill="x", padx=2)
    
    ttk.Button(
        btns_frame,
        image=icon_toggle_stretch,
        compound="left",
        text="Auto Stretch",
        command=lambda: viewer_instance and viewer_instance.toggle_auto_stretch()
    ).pack(fill="x", pady=2)
    
    ttk.Button(
        btns_frame,
        image=icon_toggle_log,
        compound="left",
        text="Log Hist",
        command=lambda: viewer_instance and viewer_instance.toggle_log()
    ).pack(fill="x", pady=2)
    
    ttk.Button(
        btns_frame,
        image=icon_toggle_sources,
        compound="left",
        text="Sources",
        command=lambda: viewer_instance and viewer_instance.toggle_sources()
    ).pack(fill="x", pady=2)
    
    # Keep references to icon PhotoImages so they don't get garbage-collected
    root.icon_browse        = icon_browse
    root.icon_start         = icon_start
    root.icon_open_fits     = icon_open_fits
    root.icon_zoom_in       = icon_zoom_in
    root.icon_zoom_out      = icon_zoom_out
    root.icon_reset         = icon_reset
    root.icon_toggle_stretch= icon_toggle_stretch
    root.icon_toggle_log    = icon_toggle_log
    root.icon_toggle_sources= icon_toggle_sources

    root.mainloop()

if __name__ == "__main__":
    run_app()
