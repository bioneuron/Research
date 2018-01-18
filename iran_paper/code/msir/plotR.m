
% main.m
% "/home/rasoul/software/matlab2006a/bin/matlab" -nodesktop

global climate_norm
global xdata
global itter


itter = 0;
cd '/home/rasoul/Dropbox/Programming/iran_paper/code/msir';
ADDRESS = '/home/rasoul/Dropbox/Programming/iran_paper/data.xlsx';
data1 = xlsread(ADDRESS);
INTERVAL = 45:70;
data = data1(INTERVAL,3);
%temp_new = data;

%%%%%%%%%%%%%%%%%%%% Climate Data %%%%%%%%%%%%%%%%%%%%%%
ADRS_SHUM_13 = '/home/rasoul/Dropbox/Programming/iran_paper/data/shum_2013.csv';
ADRS_SHUM_14 = '/home/rasoul/Dropbox/Programming/iran_paper/data/shum_2014.csv';
ADRS_SHUM_15 = '/home/rasoul/Dropbox/Programming/iran_paper/data/shum_2015.csv';

shum_13 = csvread(ADRS_SHUM_13);
shum_14 = csvread(ADRS_SHUM_14);
shum_15 = csvread(ADRS_SHUM_15);

shum = [shum_14(46:53,2); shum_15(1:18,2)];

shum =shum * 1e-5 + .3277;

shum_ave = mean(shum,2);
norm_shum = shum_ave;
m = 2 ./ (max(norm_shum) - min(norm_shum));
norm_shum = m * (norm_shum - min(norm_shum)) - 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ADRS_TMP_13 = '/home/rasoul/Dropbox/Programming/iran_paper/data/tmp_2013.csv';
ADRS_TMP_14 = '/home/rasoul/Dropbox/Programming/iran_paper/data/tmp_2014.csv';
ADRS_TMP_15 = '/home/rasoul/Dropbox/Programming/iran_paper/data/tmp_2015.csv';


tmp_13 = csvread(ADRS_TMP_13);
tmp_14 = csvread(ADRS_TMP_14);
tmp_15 = csvread(ADRS_TMP_15);


tmp = [tmp_14(46:53,2); tmp_15(1:18,2)];
tmp = tmp - 273;


tmp_ave = mean(tmp, 2);
norm_tmp = tmp_ave;
m = 2 ./ (max(norm_tmp)- min(norm_tmp));
norm_tmp = m * (norm_tmp - min(norm_tmp)) - 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ADRS_PRECIP_13 = '/home/rasoul/Dropbox/Programming/iran_paper/data/precip_2013.csv';
ADRS_PRECIP_14 = '/home/rasoul/Dropbox/Programming/iran_paper/data/precip_2014.csv';
ADRS_PRECIP_15 = '/home/rasoul/Dropbox/Programming/iran_paper/data/precip_2015.csv';

precip_13 = csvread(ADRS_PRECIP_13);
precip_14 = csvread(ADRS_PRECIP_14);
precip_15 = csvread(ADRS_PRECIP_15);

precip = [precip_14(46:53,2); precip_15(1:18,2)];

%precip = precip;

precip_ave = mean(precip, 2);
norm_precip = precip_ave;
m = 2 ./ ( max(norm_precip) - min(norm_precip) );
norm_precip = m * (norm_precip - min(norm_precip)) - 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clim_data = [norm_shum norm_tmp norm_precip];

xdata = 1:size(data,1);
ydata = data;
Yfun = @(x, xdata) infection(x, xdata, clim_data);
fun = @(x)sum((Y(x)-ydata).*(Y(x)-ydata));

x0 = [7000,  35, .0001, .25, 0, 0, 0];
lb = [100, 0, .000000007, .25, -1, -1, -0];
ub = [70000, 500, .7, 1, +1, +1, +0];

options = optimoptions('lsqcurvefit','Algorithm','trust-region-reflective','Display', 'iter', 'MaxFunEvals', 10000, 'MaxIter', 10^3 );
%options.FunctionTolerance = 1e-10;
[x,resnorm,residual,exitflag,output,lambda,jacobian] = lsqcurvefit(Yfun, x0, xdata, ydata, lb, ub, options);
 
y = infection(x, xdata, clim_data);
plot([y(2:end) ydata(2:end)]);
yvals = y(2:end);
xvals = ydata(2:end);

a = x;


ft = [1:.01:xdata(end)];
  shum = interp1(1: max(xdata), clim_data(:,1), ft);
  temp = interp1(1: max(xdata), clim_data(:,2), ft);
  precip = interp1(1: max(xdata), clim_data(:,3), ft);
  
  ic = [a(1), a(2)];
  opts = odeset('RelTol', 1e-2, 'AbsTol', 1e-4);
  [t,y] = ode45(@(t,x) sir(t, a, x, ft, precip, temp, shum), ft, ic, opts);


beta = a(3) * (1 - a(5) * precip - a(6) * temp);
R = (beta .* y(:,1)' / x(4));
plot(R)

% fid2 = fopen('Result/R_4.dat', 'w');
%  for i=1:numel(R)
%      fprintf(fid2, '%f %f \n', i, R(i) );
%  end
%fclose(fid2);