# -*- coding: utf-8 -*-
"""Data Profiling.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1edFQLYihShzsrJrxlEK9ajzw4sJU8gZH
"""

!pip install ydata_profiling



"""## Reading the Dataset"""

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

cars_df = pd.read_csv( "https://drive.google.com/uc?export=download&id=10ABViLN4Q7vgIlLvepCduU4B3C6BneJR" )



"""## Creating Data Profile"""

profile = ProfileReport(cars_df, title="Pandas Profiling Report")

profile.to_notebook_iframe()

"""## Exporting the report to a file

"""

profile.to_file("UsedCar_Data_Profile.html")