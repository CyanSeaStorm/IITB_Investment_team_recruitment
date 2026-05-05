Signal logic
Buy when:
RSI < L (oversold)
Volume > rolling volume threshold
Sell when:
RSI > H
Volume > threshold

So parameters are:

L = buy threshold (RSI low)
H = sell threshold (RSI high)
W = volume quantile filter strength 

Pipeline structure 
(Finding the parameters by grid search method L, H and W by given the data.csv file. 
Searching the Parametes  ( L, H and W) by grid search method ( Cons: Computationally Expensive and time taking, other methods like Random Grid search and Bayesian Optimizatiomn I think exist did not
go in too much depth).

Coarse Grid Search (exploration) 
(Try to find the a COARSE optimization point.

Fine Grid Seach 
(Reduce the step size and try to find a better point of optimum by grid search method).  
Thing use for Optimization 

multivariate objective  = w1(Return) + w2(Sharpe Ratio) + w3(Calmer Ratio)   

The Sharpe ratio is a financial metric used to evaluate an investment's return relative to its risk (volatility), 

<img width="240" height="56" alt="image" src="https://github.com/user-attachments/assets/0aadc6a2-b30a-4156-9cbb-bdf3c387d38c" />
Rp :Expected Portfolio Return
Rf: Risk-Free Rate 
sigmap : : Standard Deviation of Portfolio Return (Volatility)

Calmar Ratio
<img width="351" height="68" alt="image" src="https://github.com/user-attachments/assets/819d39a7-644d-4415-a193-00cc7cf91b99" />
Purpose: It helps investors determine if an investment's returns justify the risk of severe losses, rather than just daily volatility.

 
here things in the functions are put after normalizing them such that they have similar value for the and no particular parameter is shown heavy bias.  

w1 = 0.3 
w2 = 0.4 
w3 - 0.3  


(These things I chose, there are other ways to determine them,I had researched them previously, like by finding the best w1 and w2 and w3 then finding a point a little more equidistant) 

Then ran the backtesting over some 15 stokes to check for overfitting, preformed  well.

<img width="1202" height="827" alt="image" src="https://github.com/user-attachments/assets/16ce4c0d-66a6-44dc-942c-7fee8471e1b2" /> 
I think I could have gone for a more diverse set of companies and then implemented it on them.  
I am also a little spectical about H it is coming out to be 83, like there were cases where-in the I was able to get 72 or 72 but then I I saw that the point of optimum was a more or less at the edge of the 
heatmap and I was the Gemini to explore the it even further and the point of optimum was coming out be a better and gave a better result.  
As far as the test set is concerned should have used a little more diverse set of companies. 

A little disatified with sharpie value will have look at the same, and check if the implementation is properly done or not.
Use those parameters and back test them. 

Relevant things are attached with this code file like vbt webpages and heat maps.
This was the case where I had pushed for the H's upper limit that high as it provided an appropirate parameter according to me as it gave ratio of returns over the risks taken <img width="1257" height="825" alt="image" src="https://github.com/user-attachments/assets/8879aaab-be48-46a3-8fe7-ec5d43e54265" />, code for the same is in inital_optimize.py. this heat map motivated me to explore beyond the 79 parameter <img width="1134" height="899" alt="image" src="https://github.com/user-attachments/assets/741c33e8-f212-4351-a4a9-2c8f75946680" />, I could see that the score was increasing towards the same, I had a similar map with . 


These are the results when H had an higher limit of 73 <img width="1248" height="842" alt="image" src="https://github.com/user-attachments/assets/68761427-7e3a-4f32-b601-3d5956832f63" />
Heat map for the same.
<img width="2400" height="1800" alt="fine_grid_heatmap" src="https://github.com/user-attachments/assets/29a0b9df-8afc-429d-9ed0-82799232a60a" />. course search heat map <img width="1130" height="912" alt="image" src="https://github.com/user-attachments/assets/f39aace1-0227-406d-a09b-db609ab3bfd0" />

The code was generated with the help of gemini.


