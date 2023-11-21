import xarray as xr

print("begin")
data_nc = xr.open_mfdataset('/Users/sav1tr/Documents/school/PGM/dataset-new/*.nc')
print("loaded all netcdf files into one dataset")
data = data_nc.to_dataframe()
print("loaded into pandas df")
data.to_csv('dataset-kerala.csv')
print("saved to csv file")
print("end")
