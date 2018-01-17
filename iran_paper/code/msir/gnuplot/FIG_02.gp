set terminal postscript color enhanced "Helvetica" 14
set output "FIG_02.eps"


# --- Graph 1
set label "RMSE = 8.57, AIC = 118.69" at 11,57
set label "RMSE = 8.10, AIC = 117.73" at 11,55
set label "RMSE = 8.43, AIC = 119.87" at 11,53
set label "RMSE = 6.45, AIC = 107.91" at 11,51
set key top right
set grid
set yrange [0:60]
plot     'data_20.dat' using 1:2 with points pt 5 lw 10 t 'REAL DATA',  \
         'data_20.dat' using 1:3 with lines lt 4 lw 2 t 'SIR', \
         'data_21.dat' using 1:3 with lines lt 2 lw 2 t 'SIR+P', \
	 'data_22.dat' using 1:3 with lines lt 3 lw 2 t 'SIR+T', \
	 'data_24.dat' using 1:3 with lines lt 7 lw 2 t 'SIR+P+T'
	

