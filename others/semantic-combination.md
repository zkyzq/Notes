# 学习语义组合p=f(a,b)

1.线性组合: p=W[a;b],包括sum，weighted average
2.增加非线性激活函数g（sigmoid ortanh），the nonlinearity al-lows to express a wider range of functions。
缺点：it is almost certainly too much to expect a single fixed matrix W to be able to capture the meaning combination effects 
of all natural language operators. Afterall, inside the functiong, we have the same lineartransformation for all possible 
pairs of word vectors.

3. 一个是矩阵，一个是向量：f(extremely strong) = A(extremely) * b(strong)
缺点：capture linear functions only for pairs of words whereas we would like nonlinear functions to compute compositional 
meaning representations for multi-word phrases.

4. 结合123优点，每个item的表达有向量和矩阵两部分(Matrix-Vector)，p=g(W[Ba;Ab])。
举例：extremely（a可以接近零向量），strong（B可以接近单位矩阵）
写作手法高：先介绍自己，再说This function builds upon and generalizes several recent models in the literature，引出联系
递归recursively的方式从2个词扩展的多个词（Semantic Compositionality through Recursive Matrix-Vector Spaces）
缺点：parameters becomes very large and depends onthe size of the vocabulary

5. 改进4的缺点，用一个tensor表示combination的所有参数