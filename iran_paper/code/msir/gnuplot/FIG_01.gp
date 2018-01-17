set terminal postscript color enhanced "Helvetica" 14
set output "FIG_01.eps"


# --- Graph 1
set label "RMSE = 8.76, AIC = 98.12" at 11,57
set label "RMSE = 8.09, AIC = 96.81" at 11,55
set label "RMSE = 8.15, AIC = 97.11" at 11,53
set label "RMSE = 7.05, AIC = 93.01" at 11,51
set key top right
set grid
set yrange [0:60]
plot    'data_10.dat' using 1:2 with points pt 5 lw 10 t 'REAL DATA', \
        'data_10.dat' using 1:3 with lines lt 4 lw 2 t 'SIR', \
        'data_11.dat' using 1:3 with lines lt 2 lw 2 t 'SIR+P', \
	    'data_12.dat' using 1:3 with lines lt 3 lw 2 t 'SIR+T', \
	    'data_14.dat' using 1:3 with lines lt 7 lw 2 t 'SIR+P+T'
	


