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

