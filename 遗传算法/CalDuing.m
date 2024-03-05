function rate = CalDuing(func,data)
%CalDuing 计算适应度
%   func是一个泛化表达式(目标函数)，data是一个列向量

rate = func(data);
end