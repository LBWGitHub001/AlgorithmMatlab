function Output = CrossMutate(BinSet,Cross_rate,Mutate_rate)
%CrossMutate 进行交叉重组和变异
%   BinSet是二进制染色体,Cro.是交换率,Mut.是变异率
[num,~] = size(BinSet);
for i = 1:num
    [nums,len] = size(BinSet);
    BinSet = [BinSet;BinSet(i,1:end)];
    if rand()<=Cross_rate%发生交换
        index = ceil(rand()*nums);
        strat = ceil(rand()*len);
        temp = BinSet(index,strat:end);
        BinSet(index,strat:end) = BinSet(i+num,strat:end);
        BinSet(i+num,strat:end) = temp;
    end
    if rand()<=Mutate_rate
        index = ceil(rand()*nums);
        strat = ceil(rand()*len);
        if BinSet(index,strat) == 1
            BinSet(index,strat) = 0;
        else
            BinSet(index,strat) = 1;
        end
    end

end
Output = BinSet;
end