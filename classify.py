from src.classifier.main import ClassifierWindow
from src.utils import DatabaseHelper, SamplesHelper

database_helper = DatabaseHelper()
samples_helper = SamplesHelper()

window = ClassifierWindow(database_helper, samples_helper)
window.mainloop()
