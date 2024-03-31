data = readtable("Data.xlsx");
y_=table2array(data(2:end,2));
x_=table2array(data(1,3:end));
x_=x_.';
z_=table2array(data(2:end,3:end));
z_=-z_;
mesh(x_,y_,z_);




