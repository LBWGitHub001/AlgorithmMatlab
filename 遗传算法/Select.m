function Set = Select(BinSet,val)
%Select 自然选择
%   从输入中找到几组较大的数，将他们组成新的种群
%   其中，BinSet表示染色体，val表示他们所对应的适应度
Set = [val BinSet];%将适应度和染色体放在一起便于排序
Set = sortrows(Set,-1);

Set = Set(1:end/2,2:end);
end