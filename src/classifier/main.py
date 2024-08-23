import tkinter as tkt
import tkinter.font as tktfont
import tkinter.messagebox as tktmb
from pathlib import Path
from tkinter import Button, Tk
from tkinter.ttk import Progressbar

from PIL import Image, ImageTk

from src.utils import DatabaseHelper, SamplesHelper


class ClassifierWindow(Tk):
    def __init__(self, database_helper: DatabaseHelper, samples_helper: SamplesHelper):
        super().__init__()

        self.database_helper = database_helper
        self.samples_helper = samples_helper

        self.current_sample_filepath = None
        self.current_sample_image_tk = None  # preventing garbage collection

        self.setup_ui()
        self.update_status_frame()

        self.load_sample(self.samples_helper.samples[0])

    def setup_ui(self):
        self.title("Classify")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.frame = tkt.Frame(self)
        self.frame.grid(row=0, column=0, sticky="nwse")
        self.frame.grid_configure(padx=5, pady=5)
        self.frame.columnconfigure(0, weight=1)

        self.prop_frame = tkt.LabelFrame(self.frame, text="Properties")
        self.prop_frame.grid(row=0, column=0, sticky="we")
        self.prop_frame.columnconfigure(1, weight=1)
        tkt.Label(self.prop_frame, text="Database").grid(row=0, column=0, sticky="e")
        tkt.Label(self.prop_frame, text=repr(self.database_helper)).grid(
            row=0, column=1, sticky="w"
        )
        tkt.Label(self.prop_frame, text="Directory").grid(row=1, column=0, sticky="e")
        tkt.Label(
            self.prop_frame, text=str(self.samples_helper.directory.absolute())
        ).grid(row=1, column=1, sticky="w")

        self.status_frame = tkt.LabelFrame(self.frame, text="Status")
        self.status_frame.grid(row=1, column=0, sticky="we")
        self.status_frame.columnconfigure(1, weight=1)
        self.total_samples_count_var = tkt.StringVar(value="-")
        tkt.Label(self.status_frame, text="Total").grid(row=0, column=0, sticky="e")
        tkt.Label(self.status_frame, textvariable=self.total_samples_count_var).grid(
            row=0, column=1, sticky="w"
        )
        self.classified_samples_count_var = tkt.StringVar(value="-")
        tkt.Label(self.status_frame, text="Classified").grid(
            row=1, column=0, sticky="e"
        )
        tkt.Label(
            self.status_frame, textvariable=self.classified_samples_count_var
        ).grid(row=1, column=1, sticky="w")
        self.classify_percentage_label = tkt.Label(self.status_frame)
        self.classify_percentage_label.grid(row=2, column=0, sticky="e")
        self.classify_percentage_progressbar = Progressbar(
            self.status_frame, orient="horizontal", mode="determinate"
        )
        self.classify_percentage_progressbar.grid(row=2, column=1, sticky="we")

        self.image_filename_var = tkt.StringVar(value="Image filename")
        self.image_filename_label = tkt.Label(
            self.frame, textvariable=self.image_filename_var
        )
        self.image_filename_label.grid(row=2, column=0)

        self.image_label = tkt.Label(self.frame, text="Fill Image Here")
        self.image_label.grid(row=3, column=0, sticky="we")

        self.input_var = tkt.StringVar()
        self.input_entry_font = tktfont.Font(font="TkTextFont", size=20)
        self.input_entry = tkt.Entry(
            self.frame,
            width=25,
            textvariable=self.input_var,
            font="TkTextFont 20",
            justify="center",
        )
        self.input_entry.grid(row=4, column=0, sticky="we")
        self.input_entry.bind("<Return>", self.on_input_confirm)
        self.input_entry.bind("<grave>", self.on_input_skip)
        self.input_entry.bind("<Up>", self.on_previous_image)
        self.input_entry.bind("<Down>", self.on_next_image)

        self.actions_frame = tkt.Frame(self.frame)
        self.actions_frame.grid(row=6, column=0)

        self.previous_image_button = Button(self.actions_frame, text="< Previous")
        self.previous_image_button.grid(row=0, column=0)
        self.previous_image_button.bind("<1>", self.on_previous_image)

        self.confirm_button = Button(self.actions_frame, text="Confirm")
        self.confirm_button.grid(row=0, column=1)
        self.confirm_button.bind("<1>", self.on_input_confirm)

        self.skip_button = Button(self.actions_frame, text="Skip")
        self.skip_button.grid(row=0, column=2)
        self.skip_button.bind("<1>", self.on_input_skip)

        self.remove_button = Button(self.actions_frame, text="Remove")
        self.remove_button.grid(row=0, column=3)
        self.remove_button.bind("<1>", self.on_remove_sample)

        self.next_image_button = Button(self.actions_frame, text="Next >")
        self.next_image_button.grid(row=0, column=4)
        self.next_image_button.bind("<1>", self.on_next_image)

        self.to_unclassified_button = Button(
            self.actions_frame, text="To Unclassified >>"
        )
        self.to_unclassified_button.grid(row=0, column=5)
        self.to_unclassified_button.bind("<1>", self.skip_to_unclassified)

        self.exit_button = Button(self.frame, text="EXIT")
        self.exit_button.bind("<1>", self.on_exit)
        self.exit_button.grid(row=10, column=0)

    def update_status_frame(self):
        self.total_samples_count_var.set(str(len(self.samples_helper.samples)))
        self.classified_samples_count_var.set(str(self.database_helper.count()))

        progress = self.database_helper.count() / len(self.samples_helper.samples)
        self.classify_percentage_progressbar.config(value=progress * 100)
        self.classify_percentage_label.config(text=f"{progress:.1%}")

    def load_sample(self, sample_filepath: Path):
        image = Image.open(sample_filepath)
        height = 60
        ratio = height / image.height
        width = round(image.width * ratio)
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
        self.current_sample_image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.current_sample_image_tk)
        self.current_sample_filepath = sample_filepath
        self.image_filename_var.set(sample_filepath.name)
        label = self.database_helper.get_sample_label(sample_filepath)
        if label is not None:
            self.input_var.set(label)
            self.move_input_entry_cursor_end()

    def skip_to_unclassified(self, *args):
        for sample in self.samples_helper.samples:
            if (
                self.database_helper.get_sample_label(sample, ignore_skipped=False)
                is None
            ):
                self.load_sample(sample)
                self.clear_input_entry()
                break

    def on_exit(self, event):
        self.destroy()

    def clear_input_entry(self):
        self.input_var.set("")

    def move_input_entry_cursor_end(self):
        self.input_entry.icursor("end")

    def on_input_confirm(self, *args):
        label = self.input_var.get()

        if label == "":
            tktmb.showerror("Error", "Input cannot be empty!")
            return

        self.database_helper.classify_sample(self.current_sample_filepath, label)
        self.clear_input_entry()
        self.on_next_image()
        self.update_status_frame()

    def on_input_skip(self, *args):
        self.database_helper.skip_sample(self.current_sample_filepath)
        self.clear_input_entry()
        self.on_next_image()
        self.update_status_frame()

    def on_remove_sample(self, *args):
        self.database_helper.remove_sample(self.current_sample_filepath)

    def on_previous_image(self, *args):
        try:
            idx = self.samples_helper.samples.index(self.current_sample_filepath)
            if idx - 1 < 0:
                return
            self.clear_input_entry()
            self.load_sample(self.samples_helper.samples[idx - 1])
        except (ValueError, IndexError):
            pass

    def on_next_image(self, *args):
        try:
            idx = self.samples_helper.samples.index(self.current_sample_filepath)
            if idx + 1 >= len(self.samples_helper.samples):
                return
            self.clear_input_entry()
            self.load_sample(self.samples_helper.samples[idx + 1])
        except (ValueError, IndexError):
            pass
