# 2013/14
set terminal postscript color enhanced "Helvetica" 14
set output "R_01.eps"


# --- Graph 1
set key top righ
set grid
set title "R(t) for season 2013/14"
set yrange [0:4]
plot    'R_1.dat' using 1:2 with lines lt 1 lw 3 t 'R(SIR)', \
        'R_0.dat' using 1:2 with lines lt 2 lw 3 t 'R(SIR+P+T)'
	
