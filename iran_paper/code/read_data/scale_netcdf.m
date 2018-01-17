YEAR = 2015;
ADDRESS = '/home/rasoul/workspace/iran_paper/SHUM/shum.nc';
ncid=netcdf.open(ADDRESS,'NC_NOWRITE');


[varname vartype vardimIDs varatts] = netcdf.inqVar(ncid,5)
varid = netcdf.inqVarID(ncid,varname);
for i=1:6
    attname = netcdf.inqAttName(ncid,varid,1)
    attval = netcdf.getAtt(ncid,varid,attname)
end
gattname = netcdf.inqAttName(ncid,netcdf.getConstant('NC_GLOBAL'),1);
gattval = netcdf.getAtt(ncid,netcdf.getConstant('NC_GLOBAL'),gattname);
