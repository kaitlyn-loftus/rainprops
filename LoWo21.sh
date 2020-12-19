#!/bin/bash
echo Results for
echo "'The Physics of Falling Raindrops in Diverse Planetary Atmospheres'"
echo "Loftus and Wordsworth (202-)"
IS_CALCULATE=false
# IS_CALCULATE=true # uncomment to re-generate results 
if [[ "$IS_CALCULATE" == "true" ]]; then
    echo
    echo CALCULATIONS FOR FIGURES
    echo [this will take a long time]
    echo [like multiple hours]
    for i in {1..8}; do
        echo calculating figure ${i}
        python gen_fig0${i}.py
        echo completed figure ${i}
    done


    echo CALCULATIONS FOR TABLES
    for i in {2..3}; do
        echo calculating table ${i}
        python gen_tab0${i}.py
        echo completed table ${i}
    done

    echo CALCULATIONS FOR SI

    echo calculating figures S1-S2
    python gen_v_term.py
    echo completed figures S1-S2

    echo

    echo DONE CALCULATIONS
fi

echo
echo MAKE FIGURES
for i in {1..7}; do
    echo making figure ${i}
    python make_fig0${i}.py
    echo completed figure ${i}
done

echo
echo MAKE SFIGURES

echo making figures S1-S2
python make_v_term.py
echo completed figures S1-S2

echo making figure S3
python val_Earth_shape.py
echo completed figure S3

echo making figure S4
python val_Earth_v.py
echo completed figure S4

echo making figures S5-S6
python val_L93.py
echo completed figures S5-S6

echo making figures S7-S8
python val_G08.py
echo completed figures S7-S8