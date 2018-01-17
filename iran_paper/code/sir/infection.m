% infection.m 
 
function Y = infection(a, xdata) 

global itter
fprintf('Itter = %d\n', itter);
fprintf('Solution = (%d, %d, %d, %d) \n', a(1), a(2), a(3), ...
        a(4));
fprintf(' ... \n');
itter=itter+1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  t = [0:.01:xdata(end)];
  ic = [a(1), a(2)];
  opts = odeset('RelTol', 1e-2, 'AbsTol', 1e-4);
  [t,y] = ode45(@(t,x) sir(t, a, x), t, ic, opts);
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  Y = zeros(max(xdata), 1);
  for i = (2):(max(xdata))
      %[t,tspan]
      alpha = find(t <= i - 1);
      beta = find(t < i);
      Dt = diff(t);
%      dt = Dt(1);
      %Y(i) = a(5) * dt * sum(y((alpha(end):beta(end)),2));
      Y(i) = y(alpha(end),1) +  y(alpha(end),2) - y(beta(end),1) - y(beta(end),2);
  end
  %U = interp1(t, y(:,2), x);
  
 
  
  
  