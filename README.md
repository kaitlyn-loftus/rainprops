# RAIN PROPS
(**Prop**ertie**s** of **rain**drops.)

Code for "The Physics of Falling Raindrops in Diverse Planetary Atmospheres" by Loftus & Wordsworth. (Henceforth LoWo21.)

Core model is within the src/ directory.
Main article figures and tables labelled by number.
Supplemental material figures and table labelled by content.

Run ``LoWo21.sh`` to reproduce all results, figures, and tables from LoWo21. This script just systematically runs the assorted python files. It assumes ``python`` command will use python 3. Adjust ``IS_CALCULATE`` depending on whether you'd like to re-run all calculations. If yes, be forewarned the code will take a few hours to run everything as it is not overly optimized for speed. If no, do not modify internal file structure. (Note that subplot panel labelling was done external to python scripts, so generated figures will be missing panel letters.)
