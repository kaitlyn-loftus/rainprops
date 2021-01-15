# RAIN PROPS
(**Prop**ertie**s** of **rain**drops.)

Code associated with "The Physics of Falling Raindrops in Diverse Planetary Atmospheres" by Loftus & Wordsworth. (Henceforth LoWo21.) All code by Kaitlyn Loftus.

Core model is within``src/`` directory.
Main article figures and tables labelled by number.
Supplemental material figures and table labelled by content.

Run ``LoWo21.sh`` to reproduce all results, figures, and tables from LoWo21. This bash script just systematically runs the assorted python files. It assumes that the ``python`` command will use python 3 and that the user has not modified the repo's internal file structure. Adjust ``IS_CALCULATE`` depending on whether you'd like to re-run all calculations. If yes, be forewarned the code will take a few hours to run everything as it is not overly optimized for speed. If no, the code should take a minute or two to run. (Note that subplot panel labelling wasn't done within the python scripts, so generated figures will be missing panel letters.)

The raindrop methods described in LoWo21 sections 2-3 are implemented within ``src/`` and are intended for flexible use beyond the exact calculations performed in the paper. Ideally how to use the included functions and classes should be clear from the documentation and examples within ``gen*`` files, but the perplexed non-Kait user can direct questions to Kait at [kloftus@g.harvard.edu](mailto:kloftus@g.harvard.edu).
