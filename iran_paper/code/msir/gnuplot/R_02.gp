# 2014/15
set terminal postscript color enhanced "Helvetica" 14
set output "R_02.eps"


# --- Graph 1
set key top right
set grid
set title "R(t) for season 2014/15"

set yrange [.5:1.4]
plot    'R_3.dat' using 1:2 with lines lt 1 lw 3 t 'R(SIR)', \
        'R_2.dat' using 1:2 with lines lt 2 lw 3 t 'R(SIR+P+T)'
	