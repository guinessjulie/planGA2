import tkinter as tk
from tkinter import ttk


def generate_action():
    # Define what happens when the Generate button is clicked
    print("Generate button clicked")
    # Here you can add code to retrieve the values from the entries and process them


def create_gui():
    # Create main window
    root = tk.Tk()
    root.title("Option INI GUI")

    # Set style
    style = ttk.Style()
    style.theme_use("clam")  # Use the clam theme for a modern look

    # style.configure("TFrame", background="#2c3e50")
    # style.configure("TLabel", background="#2c3e50", foreground="white", font=("Helvetica", 10))
    # style.configure("TButton", background="#3498db", foreground="white", font=("Helvetica", 10, "bold"))
    # style.configure("TEntry", fieldbackground="#ecf0f1", foreground="#2c3e50")
    font = ("Helvetica Neue", 12)
    style.configure("TFrame", background="#f0f0f0")  # Light gray background
    style.configure("TLabel", background="#f0f0f0", foreground="#333333")  # Dark gray text
    style.configure("TButton", background="#4c8bf5", foreground="black")  # Blue button with white text
    style.configure("TEntry", fieldbackground="#f0f0f0", foreground="#f0f0f0", font=font)  # White entry with dark gray text

    # Create a main frame
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(padx=10, pady=10, fill='both', expand=True)

    # Configure grid layout
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(0, weight=1)
    main_frame.rowconfigure(1, weight=1)

    # Create frames for each section in 2x2 grid
    fitness_frame = ttk.LabelFrame(main_frame, text="Fitness", padding="10")
    fitness_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    mass_frame = ttk.LabelFrame(main_frame, text="Mass", padding="10")
    mass_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

    gaparams_frame = ttk.LabelFrame(main_frame, text="GAParams", padding="10")
    gaparams_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

    optional_frame = ttk.LabelFrame(main_frame, text="Optional", padding="10")
    optional_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

    # Fitness section
    ttk.Label(fitness_frame, text="wall_list:").grid(row=0, column=0, sticky='w')
    wall_list_entry = ttk.Entry(fitness_frame)
    wall_list_entry.grid(row=0, column=1, pady=5)
    wall_list_entry.insert(0, "south, west, east")

    ttk.Label(fitness_frame, text="road_side:").grid(row=1, column=0, sticky='w')
    road_side_entry = ttk.Entry(fitness_frame)
    road_side_entry.grid(row=1, column=1, pady=5)
    road_side_entry.insert(0, "south")

    ttk.Label(fitness_frame, text="adjacent_land:").grid(row=2, column=0, sticky='w')
    adjacent_land_entry = ttk.Entry(fitness_frame)
    adjacent_land_entry.grid(row=2, column=1, pady=5)
    adjacent_land_entry.insert(0, "east, west")

    ttk.Label(fitness_frame, text="optimal_aspect_ratio:").grid(row=3, column=0, sticky='w')
    optimal_aspect_ratio_entry = ttk.Entry(fitness_frame)
    optimal_aspect_ratio_entry.grid(row=3, column=1, pady=5)
    optimal_aspect_ratio_entry.insert(0, "golden_ratio")

    ttk.Label(fitness_frame, text="fitoption:").grid(row=4, column=0, sticky='w')
    fitoption_entry = ttk.Entry(fitness_frame)
    fitoption_entry.grid(row=4, column=1, pady=5)
    fitoption_entry.insert(0, "f(SSR), f(PAR), f(VSymm), f(FSH)")

    # Mass section
    ttk.Label(mass_frame, text="width:").grid(row=0, column=0, sticky='w')
    width_entry = ttk.Entry(mass_frame)
    width_entry.grid(row=0, column=1, pady=5)
    width_entry.insert(0, "8")

    ttk.Label(mass_frame, text="height:").grid(row=1, column=0, sticky='w')
    height_entry = ttk.Entry(mass_frame)
    height_entry.grid(row=1, column=1, pady=5)
    height_entry.insert(0, "6")

    ttk.Label(mass_frame, text="required_faratio:").grid(row=2, column=0, sticky='w')
    required_faratio_entry = ttk.Entry(mass_frame)
    required_faratio_entry.grid(row=2, column=1, pady=5)
    required_faratio_entry.insert(0, "0.3")

    ttk.Label(mass_frame, text="cell_length:").grid(row=3, column=0, sticky='w')
    cell_length_entry = ttk.Entry(mass_frame)
    cell_length_entry.grid(row=3, column=1, pady=5)
    cell_length_entry.insert(0, "3")

    ttk.Label(mass_frame, text="max_area:").grid(row=4, column=0, sticky='w')
    max_area_entry = ttk.Entry(mass_frame)
    max_area_entry.grid(row=4, column=1, pady=5)
    max_area_entry.insert(0, "150")

    ttk.Label(mass_frame, text="min_area:").grid(row=5, column=0, sticky='w')
    min_area_entry = ttk.Entry(mass_frame)
    min_area_entry.grid(row=5, column=1, pady=5)
    min_area_entry.insert(0, "120")

    ttk.Label(mass_frame, text="adjacent_distance_south:").grid(row=6, column=0, sticky='w')
    adjacent_distance_south_entry = ttk.Entry(mass_frame)
    adjacent_distance_south_entry.grid(row=6, column=1, pady=5)
    adjacent_distance_south_entry.insert(0, "2")

    ttk.Label(mass_frame, text="road_distance:").grid(row=7, column=0, sticky='w')
    road_distance_entry = ttk.Entry(mass_frame)
    road_distance_entry.grid(row=7, column=1, pady=5)
    road_distance_entry.insert(0, "2")

    ttk.Label(mass_frame, text="road_width:").grid(row=8, column=0, sticky='w')
    road_width_entry = ttk.Entry(mass_frame)
    road_width_entry.grid(row=8, column=1, pady=5)
    road_width_entry.insert(0, "4")

    ttk.Label(mass_frame, text="height_diff_south:").grid(row=9, column=0, sticky='w')
    height_diff_south_entry = ttk.Entry(mass_frame)
    height_diff_south_entry.grid(row=9, column=1, pady=5)
    height_diff_south_entry.insert(0, "15")

    ttk.Label(mass_frame, text="south_gap:").grid(row=10, column=0, sticky='w')
    south_gap_entry = ttk.Entry(mass_frame)
    south_gap_entry.grid(row=10, column=1, pady=5)
    south_gap_entry.insert(0, "1")

    ttk.Label(mass_frame, text="north_gap:").grid(row=11, column=0, sticky='w')
    north_gap_entry = ttk.Entry(mass_frame)
    north_gap_entry.grid(row=11, column=1, pady=5)
    north_gap_entry.insert(0, "1")

    ttk.Label(mass_frame, text="west_gap:").grid(row=12, column=0, sticky='w')
    west_gap_entry = ttk.Entry(mass_frame)
    west_gap_entry.grid(row=12, column=1, pady=5)
    west_gap_entry.insert(0, "1")

    ttk.Label(mass_frame, text="east_gap:").grid(row=13, column=0, sticky='w')
    east_gap_entry = ttk.Entry(mass_frame)
    east_gap_entry.grid(row=13, column=1, pady=5)
    east_gap_entry.insert(0, "1")

    ttk.Label(mass_frame, text="wall_height:").grid(row=14, column=0, sticky='w')
    wall_height_entry = ttk.Entry(mass_frame)
    wall_height_entry.grid(row=14, column=1, pady=5)
    wall_height_entry.insert(0, "2")

    ttk.Label(mass_frame, text="setback_requirement:").grid(row=15, column=0, sticky='w')
    setback_requirement_entry = ttk.Entry(mass_frame)
    setback_requirement_entry.grid(row=15, column=1, pady=5)
    setback_requirement_entry.insert(0, "1")

    ttk.Label(mass_frame, text="numfig:").grid(row=16, column=0, sticky='w')
    numfig_entry = ttk.Entry(mass_frame)
    numfig_entry.grid(row=16, column=1, pady=5)
    numfig_entry.insert(0, "50")

    ttk.Label(mass_frame, text="numcell:").grid(row=17, column=0, sticky='w')
    numcell_entry = ttk.Entry(mass_frame)
    numcell_entry.grid(row=17, column=1, pady=5)
    numcell_entry.insert(0, "16")

    # GAParams section
    ttk.Label(gaparams_frame, text="expandsize:").grid(row=0, column=0, sticky='w')
    expandsize_entry = ttk.Entry(gaparams_frame)
    expandsize_entry.grid(row=0, column=1, pady=5)
    expandsize_entry.insert(0, "1")

    ttk.Label(gaparams_frame, text="numcross:").grid(row=1, column=0, sticky='w')
    numcross_entry = ttk.Entry(gaparams_frame)
    numcross_entry.grid(row=1, column=1, pady=5)
    numcross_entry.insert(0, "10")

    ttk.Label(gaparams_frame, text="mutationrate:").grid(row=2, column=0, sticky='w')
    mutationrate_entry = ttk.Entry(gaparams_frame)
    mutationrate_entry.grid(row=2, column=1, pady=5)
    mutationrate_entry.insert(0, "0.1")

    ttk.Label(gaparams_frame, text="ngeneration:").grid(row=3, column=0, sticky='w')
    ngeneration_entry = ttk.Entry(gaparams_frame)
    ngeneration_entry.grid(row=3, column=1, pady=5)
    ngeneration_entry.insert(0, "1")

    ttk.Label(gaparams_frame, text="population_size:").grid(row=4, column=0, sticky='w')
    population_size_entry = ttk.Entry(gaparams_frame)
    population_size_entry.grid(row=4, column=1, pady=5)
    population_size_entry.insert(0, "500")

    ttk.Label(gaparams_frame, text="crossoverchance:").grid(row=5, column=0, sticky='w')
    crossoverchance_entry = ttk.Entry(gaparams_frame)
    crossoverchance_entry.grid(row=5, column=1, pady=5)
    crossoverchance_entry.insert(0, "1.0")

    ttk.Label(gaparams_frame, text="dnalength:").grid(row=6, column=0, sticky='w')
    dnalength_entry = ttk.Entry(gaparams_frame)
    dnalength_entry.grid(row=6, column=1, pady=5)
    dnalength_entry.insert(0, "0")

    ttk.Label(gaparams_frame, text="matingpool_multiple:").grid(row=7, column=0, sticky='w')
    matingpool_multiple_entry = ttk.Entry(gaparams_frame)
    matingpool_multiple_entry.grid(row=7, column=1, pady=5)
    matingpool_multiple_entry.insert(0, "4")

    ttk.Label(gaparams_frame, text="keep_best_rate:").grid(row=8, column=0, sticky='w')
    keep_best_rate_entry = ttk.Entry(gaparams_frame)
    keep_best_rate_entry.grid(row=8, column=1, pady=5)
    keep_best_rate_entry.insert(0, "0.1")

    ttk.Label(gaparams_frame, text="max_fitness:").grid(row=9, column=0, sticky='w')
    max_fitness_entry = ttk.Entry(gaparams_frame)
    max_fitness_entry.grid(row=9, column=1, pady=5)
    max_fitness_entry.insert(0, "0.7")

    # Optional section
    ttk.Label(optional_frame, text="forcednalength:").grid(row=0, column=0, sticky='w')
    forcednalength_entry = ttk.Entry(optional_frame)
    forcednalength_entry.grid(row=0, column=1, pady=5)
    forcednalength_entry.insert(0, "yes")

    ttk.Label(optional_frame, text="selection:").grid(row=1, column=0, sticky='w')
    selection_entry = ttk.Entry(optional_frame)
    selection_entry.grid(row=1, column=1, pady=5)
    selection_entry.insert(0, "roulette")

    ttk.Label(optional_frame, text="start_position:").grid(row=2, column=0, sticky='w')
    start_position_entry = ttk.Entry(optional_frame)
    start_position_entry.grid(row=2, column=1, pady=5)
    start_position_entry.insert(0, "random")

    ttk.Label(optional_frame, text="mutate_option:").grid(row=3, column=0, sticky='w')
    mutate_option_entry = ttk.Entry(optional_frame)
    mutate_option_entry.grid(row=3, column=1, pady=5)
    mutate_option_entry.insert(0, "rate")

    ttk.Label(optional_frame, text="crossover_method:").grid(row=4, column=0, sticky='w')
    crossover_method_entry = ttk.Entry(optional_frame)
    crossover_method_entry.grid(row=4, column=1, pady=5)
    crossover_method_entry.insert(0, "b")

    ttk.Label(optional_frame, text="init_method:").grid(row=5, column=0, sticky='w')
    init_method_entry = ttk.Entry(optional_frame)
    init_method_entry.grid(row=5, column=1, pady=5)
    init_method_entry.insert(0, "c")

    # Generate button
    generate_button = ttk.Button(main_frame, text="Generate", command=generate_action)
    generate_button.grid(row=2, column=0, columnspan=2, pady=10)

    # Main loop
    root.mainloop()


if __name__ == "__main__":
    create_gui()
