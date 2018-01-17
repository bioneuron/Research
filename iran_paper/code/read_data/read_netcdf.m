YEAR = 2013;
ADDRESS = '/home/rasoul/workspace/iran_paper/TMP/tmp.nc';
ncid=netcdf.open(ADDRESS,'NC_NOWRITE');

%TIME
ncvar = netcdf.inqVar(ncid,3)
varid=netcdf.inqVarID(ncid,ncvar);
time_id=netcdf.getVar(ncid,varid);
date_start = strcat('01-Jan-',num2str(YEAR));
date_end = strcat('31-Dec-',num2str(YEAR));
%start_id = daysdif('1-Jan-2000', date_start) + 1;
start_id = datenum(date_start)-datenum('1-jan-2000') + 1;
%end_id = daysdif('1-Jan-2000', date_end) + 1;
end_id = datenum(date_end)-datenum('1-jan-2000') + 1;
%week_id = weeknum(datenum(date_start):datenum(date_end));
date_vec = 1:(end_id-start_id+1);
week_id = week(datetime(date_start):datetime(date_end))

%DATA
ncvar = netcdf.inqVar(ncid,5);
varid=netcdf.inqVarID(ncid,ncvar);
data=netcdf.getVar(ncid,varid);

data_week = zeros(max(week_id), 1);
for i = 1 : max(week_id)
    id = find(week_id == i);
    data(data<-1e10)=0;
    data1 =  mean(mean(data,1),2); 
    data_week(i) = mean(data1(id + start_id - 1));
end
    
plot(data_week)
%xwrite('/home/rasoul/workspace/iran_paper/tmp_2000.xlsx',data_week)
%min(min(min(data)))

%netcdf.getAtt(ncid,3,netcdf.inqAttName(ncid,3,10))



  fid2 = fopen('/home/rasoul/workspace/iran_paper/data/tmp_2013.csv', 'w');
  for i=1:max(week_id)
      fprintf(fid2, '%d, %f \n',i, data_week(i));
  end
fclose(fid2);

