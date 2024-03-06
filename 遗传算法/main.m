clear;
clc;
%在下方输入你要求解的函数即可求得最大值
f=@(x) x + 18*cos(5*x) + 7*cos(4*x) + 17*sin(6*x);
%设置交换率和变异率两个参数
Cross_rate = 0.6;%交换率 
Mutate_rate = 0.01;%变异率 

sizeGroup = 100;%种群规模
binSet = floor(rand(sizeGroup,17)*2);
val = CalDuing(f,Encoding(binSet));
val(1)/val(2)
gate = 1.000001;
while val(1)/val(2)>=gate || val(2)/val(1)>=gate || val(1)*val(2)<=0
    val = CalDuing(f,Encoding(binSet));
    binSet = Select(binSet,val);
    binSet = CrossMutate(binSet,Cross_rate,Mutate_rate);
    x_val = Encoding(binSet);
    val = CalDuing(f,x_val);
    val(1)/val(2)
end

max = val(1)

x = 1:0.001:9;
y = f(x);
plot(x,y);
hold on;
scatter(x_val.',val.','*r');
hold on;
scatter(x_val(1),val(1),'og','filled');
legend("目标函数","最后一次迭代的染色体组","优化结果");
grid on;