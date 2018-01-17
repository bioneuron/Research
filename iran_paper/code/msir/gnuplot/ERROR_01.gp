set terminal postscript color enhanced "Helvetica" 14
set output "ERROR_01.eps"


# --- Graph 1
set key top right
set grid
set xrange [10:33]
set yrange [-40:30]
set xlabel "Week"
set ylabel "Residual Error"
plot   'data_10.dat' using 1:($3-$2) with points pt 5 lw 10 t 'RMSE(SIR)', \
       'data_14.dat' using 1:($3-$2) with points pt 1 lw 8 t 'RMSE(SIR+P+T)'
	
