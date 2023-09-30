from .icLUTgenerator import *
from .pol_icLUTgenerator import *
from .wcLUTgenerator import *
from .tools import *
from .simulation_tools import *
from .conveniences import *
from .retrieval_functions import *

# Dictionaries translating libRadtran habit names to shorts used in Thesis
name_dict = {"hollow_bullet_rosette":"HBR", 
             "column_8elements":"8-col.", 
             "droxtal":"drox.", 
             "ghm":"GHM", 
             "plate":"plate", 
             "plate_10elements":"10-plate", 
             "plate_5elements":"5-plate", 
             "solid_bullet_rosette":"SBR",
             "solid_column":"SC",
             "hollow_column":"HC"}

name_dict_rough = {"hollow_bullet_rosette":"HBR_r", 
             "column_8elements":"8-col_r", 
             "droxtal":"drox_r", 
             "ghm":"GHM", 
             "plate":"plate_r", 
             "plate_10elements":"10-plate_r", 
             "plate_5elements":"5-plate_r", 
             "solid_bullet_rosette":"SBR_r",
             "solid_column":"SC_r",
             "hollow_column":"HC_r"}

name_dict_smooth = {"hollow_bullet_rosette":"HBR_s", 
             "column_8elements":"8-col_s", 
             "droxtal":"drox_s", 
             "ghm":"GHM", 
             "plate":"plate_s", 
             "plate_10elements":"10-plate_s", 
             "plate_5elements":"5-plate_s", 
             "solid_bullet_rosette":"SBR_s",
             "solid_column":"SC_s",
             "hollow_column":"HC_s"}

single_habits = ["solid_column", "hollow_column", "plate", "droxtal"]
aggregate_habits = ["column_8elements", "plate_5elements", "plate_10elements"]
bullet_rosette_habits = ["solid_bullet_rosette", "hollow_bullet_rosette"]

plate_habits = ["plate", "plate_5elements", "plate_10elements"]
column_habits = ["solid_column", "hollow_column", "column_8elements"]
bullet_rosette_and_droxtal_habits = ["solid_bullet_rosette", "hollow_bullet_rosette", "droxtal"]