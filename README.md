# 2019kdxf

一些思考：

1.特征稳定性检验，每个特征在训练和测试都应该分布相同。但是，在最后的入模特征中未检查特征分布的稳定性。应该用画图或者计算psi等方式，检查所有变量的稳定性。  
2.关于时间，时间当特征可能有用，可能没有用，需要实际跑模型看看。类似，样本权重。  
3.关于嫁接，如果分为A和B数据，记得优先使用全部数据试试，可能带来意想不到的效果。具体的嫁接方式可以参考砍手豪大佬的知乎文章https://zhuanlan.zhihu.com/p/51901122  
4.关于模型，相同数据和特征下，不同模型的结果可能差异很大，需要不断尝试。一个模型真的可以甩其他模型几条街。  
5.关于数据量，当数据量特别大的时候，可以进行采样，规则过滤等等，但是提取特征需要在此之前进行，否则可能会损失很多信息。如果可以感觉使用NN应该是不错的。  
6.关于特征重要性，特征重要性真的是玄学，高特征重要性可能导致过拟合，重要性低的特征删除之后效果还下降了。总之，特征重要性越高的，不一定真的有用。最好，还是做一下分布检测，相关性等等。  
7.关于模型评价标准，例如f1 score最好用oof进行下阈值搜索，找到相对最优的阈值。  
8.关于模型融合。融合大法好，至今也没真正的好好融合过，都是取的平均值附近的权重进行尝试，后面还是需要恶补一下这方面的知识和技巧。  
  
  
最后，感谢一路一起摸鱼的世界~  
