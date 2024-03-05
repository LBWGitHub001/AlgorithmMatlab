clear;
clc;
f=@(x) x + 10*sin(5*x) + 7*cos(4*x);

Cross_rate = 0.6;%交换率
Mutate_rate = 0.01;%变异率

sizeGroup = 100;%种群规模
binSet = floor(rand(sizeGroup,17)*2);
val = CalDuing(f,Encoding(binSet));
while val(1)/val(2)>=1.001 || val(1)*val(2)<=0
    val = CalDuing(f,Encoding(binSet));
    binSet = Select(binSet,val);
    binSet = CrossMutate(binSet,Cross_rate,Mutate_rate);
    val = CalDuing(f,Encoding(binSet));
    val(1)/val(2)
end

max = val(1)

x = 1:0.001:9;
y = f(x);
plot(x,y);
grid on;