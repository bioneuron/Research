% sir.m 
%
function dx = sir(t, a, x, ft, f, g, h) 
  %global beta epsilon recovery; 
  P = interp1(ft, f, t);
  T = interp1(ft, g, t);
  S = interp1(ft, h, t);
  
  dx = [0; 0]; 
  dx(1) = -a(3) * (1 - a(5) * P - a(6) * T - a(7) * S) * x(1) * x(2) ; 
  dx(2) =  a(3) * (1 - a(5) * P - a(6) * T - a(7) * S) * x(1) * x(2) - a(4) * x(2);
 