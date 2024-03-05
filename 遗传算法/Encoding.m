function Output = Encoding(binInput)
%Encoding 此处显示有关此函数的摘要
%   本函数用于二进制到十进制的解码
encoding = @(b)b*(9/(2^17-1));
[rows, cols] = size(binInput);
Output = [0];
for i = 1:rows
Dec=0;
    for j = 1:cols
        Dec = Dec+binInput(i,j)*(2^(j-1));%数据存储格式是高位在右
    end
Output =[Output;encoding(Dec)];
Dec = 0;
end
Output = Output(2:end);
end