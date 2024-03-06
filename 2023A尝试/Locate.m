Location = readtable("附件.xlsx");
Location = table2array(Location);
scatter(Location(1:end,1),Location(1:end,2),'*b');
grid on;