function gard = CalGard(InputZ,xx,yy,index)
%CalGard 计算一点处的梯度
%  InputZ是计算的x，是z值的矩阵，index是需要求解点的索引坐标
[Fx,Fy] = gradient(InputZ); %计算数值梯度
IsIn = (xx==index(1))&(yy==index(2));%检测输入的合法性，是否存在这样的一个点
dot = find(IsIn);
gard = [Fx(dot),Fy(dot)];

end