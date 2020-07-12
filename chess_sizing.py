"""
script to help match the square size of your board to the base diameter of your
king, and vice versa
"""
import tkinter as tk

# [user input] what do you want to know? ('square size', 'king diameter')
want_to_know = 'king_diameter'

# make a gui window
window = tk.Tk()
# name the window
window.title('Chess Sizing Tool')
# add an icon to the window
icon_image = tk.PhotoImage(file='king.png')
window.iconphoto(False, icon_image)
# create a menu bar
menu_bar = tk.Menu(window)
file_menu = tk.Menu(menu_bar, tearoff=False)
file_menu.add_command(label='Exit', command=window.destroy)
menu_bar.add_cascade(label='File', menu=file_menu)
window.config(menu=menu_bar)
# add a label prompting the drop-down
label_known = tk.Label(text='I know the...', width=20, height=5)
label_known.grid(row=0, column=0)
# create a drop-down list for what is known
drop_down_options = ['king diameter', 'square size']
str_variable = tk.StringVar(window)
str_variable.set(drop_down_options[0])
drop_down_menu = tk.OptionMenu(window, str_variable, *drop_down_options)
drop_down_menu.config(width=2*len(max(drop_down_options)))
drop_down_menu.grid(row=0, column=1)
# add a padding label to set the width
label_pad = tk.Label(width=20, height=5)
label_pad.grid(row=0, column=2)
# add a label prompting the measurement
label_measurement = tk.Label(text='It is...', width=10, height=5)
label_measurement.grid(row=1, column=0)
# create an entry field
entry_measurement = tk.Entry(width=10)
entry_measurement.grid(row=1, column=1)
# create a drop-down list for the units
drop_down_options = ['inches', 'centimeters']
unit_str_var = tk.StringVar(window)
unit_str_var.set(drop_down_options[0])
drop_down_units = tk.OptionMenu(window, unit_str_var, *drop_down_options)
drop_down_units.config(width=2*len(max(drop_down_options)))
drop_down_units.grid(row=1, column=3)
# create a button
button_calc = tk.Button(text='Calculate', width=40, height=5)
button_calc.grid(row=2, columnspan=3, sticky='ew')


# force the entry box to take focus
entry_measurement.focus_set()
window.focus_force()

# launch the gui
window.mainloop()
