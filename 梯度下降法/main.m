clear;
clc;

lr = 0.3;   %学习率
%目标函数
func = @(x,y) x.^3 + y.^2 - x.*y + 5.*y + 3.*x + 15*sin(x*y) + 10*cos(3*x)
%设置求解区间
x_str=-3;
x_end=3;
y_str=-3;
y_end=3;
forward_step=0.01;
x=x_str:forward_step:x_end;
y=y_str:forward_step:y_end;
[xx,yy]=meshgrid(x,y);
z=func(xx,yy);size
mesh(xx,yy,z);
hold on;

%%开始求解
point=[0.1,0.1]
grad = CalGard(z,xx,yy,point);
while abs(grad) >= 0.001
    point = point - gard .* lr;
    gard = CalGard(z,xx,yy,point);
end
disp("找到的局部最优解的值")
point
disp("最小值是")
z = func(point(1),point(2))




