% main.m

global xdata;
global norm_climate;
global itter
itter=0;
cd '/home/rasoul/Dropbox/Programming/iran_paper/code/';
ADDRESS = '/home/rasoul/Dropbox/Programming/iran_paper/data.xlsx';
data1 = xlsread(ADDRESS);
INTERVAL = 40:80;
data = data1(INTERVAL,2);
%temp_new = data;

%%%%%%%%%%%%%Climate Data%%%%%%%%%%%%%%%
ADDRESS = '/home/rasoul/Dropbox/Programming/SEIR/Matlab/data/AH.xlsx';
data2 = xlsread(ADDRESS);
hum = data2(INTERVAL,7:12)*1e-5 + .3277;
hum1 = data1(INTERVAL,5)*1e-5 + .3277;
hum_ave = mean(hum,2);
norm_climate = hum_ave;
m = 2 ./ (max(norm_climate)- min(norm_climate));
norm_climate = m * (norm_climate - min(norm_climate)) - 1;
%hum1 = zeros(size(hum));
%hum1(2:end)= diff(hum);
%norm_climate = abs(hum1)/abs(max(hum1));
%temp_new = (temp - min(temp))/(max(temp) - min(temp));
xdata = 1:size(data,1);
ydata = data;
Yfun = @(x, xdata) infection(x,xdata);
fun = @(x)sum((Y(x)-ydata).*(Y(x)-ydata));

x0 = [7000,  35, .0001,  .3];
lb = [100, 0, .000000007, .2];
ub = [70000, 500, .7, 1];

%lb = [ 500, 0, 0, 1/5, 1/3, .2, -10];
%ub = [ 7000000, 100, 500, 2, 1, 1, 10];
%x0 = [7000, 0, 1, 1, .4,  .3, 0];

options = optimoptions('lsqcurvefit','Algorithm','trust-region-reflective', ...
    'Display', 'iter', 'MaxFunEvals', 10000, 'MaxIter', 10^3);
%options.FunctionTolerance = 1e-10;
[x,resnorm,residual,exitflag,output,lambda,jacobian]  = lsqcurvefit(Yfun, x0, xdata,...
    ydata, lb, ub, options);
 
y = infection(x, xdata);
plot([y(2:end) ydata(2:end)])
yvals = y(2:end);
xvals = ydata(2:end);

 %beta = x(4) ./ (1 -  x(7)*y(:,3)- x(8)*f');
 %plot(beta.*y(:,1)/x(6))
%plot(residual2)
rmse = (sum((y(2:end) - ydata(2:end)).^2)/numel(y(2:end)))^0.5
RE = (y(2:end) - ydata(2:end))./y(2:end)

%CI = nlparci(x,residual,'jacobian',jacobian, 'alpha', .05);
%J= jacobian;
%  xCovariance = (J.'*J)\residual;

  %AIC
  RSS=sum((y(2:end)- ydata(2:end)).^2)
  AIC = numel(ydata)*log(RSS/numel(ydata)) + 16
  
  [maxval1,ind1] = max(y(2:end))
  [maxval2,ind2] = max(ydata(2:end))
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
  
  fid = fopen('result2/data_33.dat', 'w');
  for i=1:numel(xvals)
      fprintf(fid, '%f %f %f \n',10+i,  xvals(i), yvals(i) );
  end
fclose(fid);
  
 
  
  
  fid1 = fopen('result2/data_34.dat', 'w');
  
  fprintf(fid1, 'x0 = ( %f, %f, %f, %f, %f, %f, %f, %f)\n\n', x0(1), x0(2), x0(3), x0(4), x0(5), x0(6), x0(7), x0(8));
 
  fprintf(fid1, '-----------------------------------------------------------------------------\n\n');
  
 fprintf(fid1, 'x = ( %f, %f, %f, %f, %f, %f, %f, %f)\n\n', x(1), x(2), x(3), x(4), x(5), x(6), x(7), x(8));


  fprintf(fid1, 'RMSE = %f \n\n', rmse);
  
  fprintf(fid1, 'AIC = %f \n\n',  AIC);
  
  fprintf(fid1, '(Pick Time, Pick Magnitude) = (%f,%f) \n\n', ind1, maxval1);
  
  fprintf(fid1, '(Original Pick Time, Original Pick Magnitude) = (%f, %f) \n\n', ind2, maxval2);

 
fclose(fid1);  
  
%%% Genetic Algorithm
%opts = optimoptions('ga','PlotFcn',@gaplotbestf);
%Call the ga solver where x(1) has integer values
%rng(1,'twister') % for reproducibility
%IntCon = 1;
%[x,fval,exitflag] = ga(fun,length(lb),[],[],[],[],lb,ub,[],IntCon,opts);



  fid2 = fopen('result2/data_35.dat', 'w');
  for i=1:numel(t)
      fprintf(fid2, '%f %f %f %f\n',10 + t(i),  y(i,1), y(i,2), y(i,3));
  end
fclose(fid2);




%*  fid = fopen('result1/RMSE2.dat', 'w');
%for i=1:numel(xvals)
      %fprintf(fid, '%f %f\n',i+10, RE(i));
      % end
  %fclose(fid);
