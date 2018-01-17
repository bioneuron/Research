% sir.m 
%
function dx = sir(t, a, x) 
  %global beta epsilon recovery; 

  dx = [0; 0]; 
  dx(1) = -a(3)* x(2) * x(1) ; 
  dx(2) =  a(3)* x(2) * x(1) - a(4) * x(2);
 