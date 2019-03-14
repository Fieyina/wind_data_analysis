# 本项目正在（指当下的每小时）更新中
# wind_data_analysis
Wind 数据分析；通过拟合分布反推统计量，计算企业出现该情况的概率值，衡量企业异常程度
### 思路
#### 〇、将财务数据进行简单运算后，得出的财务指标（如：现金比率）称为原始数据
#### 一、对原始数据取自然对数
#### 二、分行业、分指标进行数据处理；对于每个数据集：
> 通过拟合累计分布函数的形式，得出其均值与标准差  
> 如果直接用计算的形式，会默认不存在异常点，且完全符合正态分布  
> 但实际部分行业指标仅符合正态分布的一部分
1. 令指标列升序
2. 对 `y=各个公司，x=指标` 这一数列  
@in list（x=指标，y=各个公司）  
@fitting `a * (b + special.erf((x - u) / o * 2 ** 0.5))`    
@return a，b，u，o    
@name **累计分布函数**。   
3. 对**累计分布函数**求导
在实操中可以通过，对这一函数切出散点后，对散点进行拟合
@in list(x,y)    
@fitting `a * (1 / (2 * np.pi * o ** 2) ** 0.5) * np.exp(-(x - u) ** 2 / (2 * o ** 2))`     
@return a，o，u    
@name **密度分布函数**    
4. 将原始数据的散点图（1），拟合的累计分布函数（2），拟合的密度分布函数（3）做在同一张图内  
用于备查，检查拟合程度。
> 备查图示例  
![这里随便写文字](https://github.com/Fieyina/wind_data_analysis/blob/master/lib/inventory%20ratio.png)
#### 三、对于指标有效性较低的行业指标予以剔除，对于每个行业指标：
> 判断标准即：正态分布的峰是否足够包含在整个图像中
1. 以`正态分布均值 u 处的概率密度函数的取值为 m`，m 的 3/7 为 n。
2. 将 n 代回概率密度函数求出两个对应的 x 轴点位 p，q。
3. 若 p，q 包含在数据集的上下界内，则认为指标有效。
#### 四、根据上述统计结果，计算每家企业的统计量数值
