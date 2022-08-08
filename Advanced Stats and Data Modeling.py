import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statistics as s
from scipy.stats import ttest_1samp
from matplotlib import colors
import xlwings as xw

AthletesDF = pd.read_excel('2018-19 College Basketball Athletes.xlsx',
                            skiprows=(0,1), header=None,
                            names=['LastName','FirstName','College','Region',
                            'StateTheyPlayIn','PublicOrPrivate', 'Height', 'Assists', 'Blocks',
                            'Rebounds', 'Points', 'NumberofGamesPlayed'],
                            usecols='A:L')

SchoolsDF = pd.read_excel('Data for Top 50 Colleges.xlsx',
                          skiprows = (1), header = None,
                          names=['Name', 'PublicOrPrivate', 'Region', 'Ivy',
                                 'Age', 'Tuition', 'AvgClassSize', 'AvgStudentAge',
                                 'AcceptanceRate', 'AvgACTScore', 'AvgHighSchoolGPA'],
                          usecols='A:K')

print(SchoolsDF)
print(AthletesDF)

# First Regression Analysis Using Sheet1

print ("First Regression Analysis Using College Basketball Athletes Height and Rebounds")
xvar='Height'
yvar='Rebounds'
x=AthletesDF[xvar]
y=AthletesDF[yvar]

slope, intercept, r_value, p_val, std_err = stats.linregress(y=y,x=x)
print("This is regression with Ho: X does not help to predict Y/The slope is 0")

if np.sign(slope) < 0:   
    slsign = ""
else:
    slsign = "+"

regeq = f"{yvar} = {round(intercept,3)} {slsign} {round(slope,3)}{xvar}"

print(f"The equation is {regeq}")

print(f"The R-Squared is {round(r_value**2,4)} and the p-value is {round(p_val,4)}")

alpha=.05
if p_val < alpha:
    print("Conclusion: Reject Ho: X does help predict Y")
else:
    print("Conclusion: Fail to Reject Ho: We can't reject that X doesn't help to predict Y")

plt.scatter(x,y,color='black')
xyCorr = round(x.corr(y),3)
plt.suptitle(f"Correlation: {xyCorr}  R-Squared: {round(r_value**2,4)} p-value: {round(p_val,4)}")
plt.title(regeq, size=10)
predict_y = intercept + slope * x
plt.plot(x,predict_y, 'r-')
plt.xlabel(xvar)
plt.ylabel(yvar)
plt.savefig('RegressionOneX.png', bbox_inches='tight')
plt.show()

# Second Regression Analysis Using Sheet2
print ("Second Regression Analysis Using College Basketball Athletes Assists and Height")
xvar='Assists'
yvar='Height'
x=AthletesDF[xvar]
y=AthletesDF[yvar]

slope, intercept, r_value, p_val, std_err = stats.linregress(y=y,x=x)
print("This is regression with Ho: X does not help to predict Y/The slope is 0")

if np.sign(slope) < 0:   
    slsign = ""
else:
    slsign = "+"

regeq = f"{yvar} = {round(intercept,3)} {slsign} {round(slope,3)}{xvar}"

print(f"The equation is {regeq}")

print(f"The R-Squared is {round(r_value**2,4)} and the p-value is {round(p_val,4)}")

alpha=.05
if p_val < alpha:
    print("Conclusion: Reject Ho: X does help predict Y")
else:
    print("Conclusion: Fail to Reject Ho: We can't reject that X doesn't help to predict Y")

plt.scatter(x,y,color='black')
xyCorr = round(x.corr(y),3)
plt.suptitle(f"Correlation: {xyCorr}  R-Squared: {round(r_value**2,4)} p-value: {round(p_val,4)}")
plt.title(regeq, size=10)
predict_y = intercept + slope * x
plt.plot(x,predict_y, 'r-')
plt.xlabel(xvar)
plt.ylabel(yvar)
plt.savefig('RegressionOneX.png', bbox_inches='tight')
plt.show()

# Third Regression analysis using sheet2
print ("Third Regression Analysis Using Colleges Avg ACT Score and Tuition")
xvar='AvgACTScore'
yvar='Tuition'
x=SchoolsDF[xvar]
y=SchoolsDF[yvar]

slope, intercept, r_value, p_val, std_err = stats.linregress(y=y,x=x)
print("This is regression with Ho: X does not help to predict Y/The slope is 0")

if np.sign(slope) < 0:   
    slsign = ""
else:
    slsign = "+"

regeq = f"{yvar} = {round(intercept,3)} {slsign} {round(slope,3)}{xvar}"

print(f"The equation is {regeq}")

print(f"The R-Squared is {round(r_value**2,4)} and the p-value is {round(p_val,4)}")

alpha=.05
if p_val < alpha:
    print("Conclusion: Reject Ho: X does help predict Y")
else:
    print("Conclusion: Fail to Reject Ho: We can't reject that X doesn't help to predict Y")

plt.scatter(x,y,color='black')
xyCorr = round(x.corr(y),3)
plt.suptitle(f"Correlation: {xyCorr}  R-Squared: {round(r_value**2,4)} p-value: {round(p_val,4)}")
plt.title(regeq, size=10)
predict_y = intercept + slope * x
plt.plot(x,predict_y, 'r-')
plt.xlabel(xvar)
plt.ylabel(yvar)
plt.savefig('RegressionOneX.png', bbox_inches='tight')
plt.show()

# Forth Regression analysis using sheet2
print ("Forth Regression Analysis Using Colleges Acceptance Rate and Avg Class Size")
xvar='AcceptanceRate'
yvar='AvgClassSize'
x=SchoolsDF[xvar]
y=SchoolsDF[yvar]

slope, intercept, r_value, p_val, std_err = stats.linregress(y=y,x=x)
print("This is regression with Ho: X does not help to predict Y/The slope is 0")

if np.sign(slope) < 0:   
    slsign = ""
else:
    slsign = "+"

regeq = f"{yvar} = {round(intercept,3)} {slsign} {round(slope,3)}{xvar}"

print(f"The equation is {regeq}")

print(f"The R-Squared is {round(r_value**2,4)} and the p-value is {round(p_val,4)}")

alpha=.05
if p_val < alpha:
    print("Conclusion: Reject Ho: X does help predict Y")
else:
    print("Conclusion: Fail to Reject Ho: We can't reject that X doesn't help to predict Y")

plt.scatter(x,y,color='black')
xyCorr = round(x.corr(y),3)
plt.suptitle(f"Correlation: {xyCorr}  R-Squared: {round(r_value**2,4)} p-value: {round(p_val,4)}")
plt.title(regeq, size=10)
predict_y = intercept + slope * x
plt.plot(x,predict_y, 'r-')
plt.xlabel(xvar)
plt.ylabel(yvar)
plt.savefig('RegressionOneX.png', bbox_inches='tight')
plt.show()

# Upper Tail Test

PopMean = 78
n       = len(AthletesDF)
xbar    = AthletesDF.Height.mean()
s       = AthletesDF.Height.std()
LoC     = 0.95
Alpha   = 1 - LoC

tscore, pvalue = ttest_1samp(AthletesDF.Height, popmean=PopMean)
print("MODEL DATA - UPPER TAIL TEST:")
print("-----------------------------")
print("Population Mean: ", PopMean)
print("Sample Size: ", n)
print("Sample Mean: ", xbar)
print("Sample standard deviation: ", s)
print("Level of Confidence: ", LoC)
print("Alpha: ", Alpha)
print()
print()
print("MODEL OUTPUT:")
print("-------------")
print("t Statistic: ", tscore)  
print("Raw P Value: ", pvalue)
Adj_pvalue = pvalue/2
print("Adj P Value: ", Adj_pvalue)
if Adj_pvalue < Alpha:
    print("Reject Ho: Our Althletes are Taller")
else:
    print("Do Not Reject Ho: Cannot conclude our Althletes are Taller")

print()
print()

#Lower Tail Test

PopMean = 2
n       = len(AthletesDF)
xbar    = AthletesDF.Assists.mean()
s       = AthletesDF.Assists.std()
LoC     = 0.95
Alpha   = 1 - LoC

tscore, pvalue = ttest_1samp(AthletesDF.Assists, popmean=PopMean)
print("MODEL DATA - LOWER TAIL TEST:")
print("-----------------------------")
print("Population Mean: ", PopMean)
print("Sample Size: ", n)
print("Sample Mean: ", xbar)
print("Sample standard deviation: ", s)
print("Level of Confidence: ", LoC)
print("Alpha: ", Alpha)
print()
print()
print("MODEL OUTPUT:")
print("-------------")
print("t Statistic: ", tscore)  
print("Raw P Value: ", pvalue)
Adj_pvalue = pvalue/2
print("Adj P Value: ", Adj_pvalue)
if Adj_pvalue < Alpha:
    print("Reject Ho: Our athletes are averaging less assists")
else:
    print("Do Not Reject Ho: Cannot conclude our athletes are averaging less assists")

print()
print()

# Two Tail Test

PopMean = 32
n       = len(AthletesDF)
xbar    = AthletesDF.NumberofGamesPlayed .mean()
s       = AthletesDF.NumberofGamesPlayed .std()
LoC     = 0.95
Alpha   = 1 - LoC

tscore, pvalue = ttest_1samp(AthletesDF.NumberofGamesPlayed, popmean=PopMean)
print("MODEL DATA - TWO TAIL TEST:")
print("-----------------------------")
print("Population Mean: ", PopMean)
print("Sample Size: ", n)
print("Sample Mean: ", xbar)
print("Sample standard deviation: ", s)
print("Level of Confidence: ", LoC)
print("Alpha: ", Alpha)
print()
print()
print("MODEL OUTPUT:")
print("-------------")
print("t Statistic: ", tscore)  
print("Raw P Value: ", pvalue)
if pvalue < Alpha:
    print("Reject Ho: Our athletes play a number of games per season not consistent with everyone else")
else:
    print("Do Not Reject Ho: Cannot conclude our athletes play a different number of games per season")
print()
print()

# Anova 
print("ANOVA TEST")
Northeast = AthletesDF[AthletesDF['Region']=='Northeast']
South = AthletesDF[AthletesDF['Region']=='South']
Midwest  = AthletesDF[AthletesDF['Region']=='Midwest']
West  = AthletesDF[AthletesDF['Region']=='West']

alpha = .2
f, p_val = stats.f_oneway(Northeast['Points'],South['Points'],Midwest['Points'],West['Points'])

print("This is a test of equal means")
print("Ho: The means of all groups are equal")
print("Ha: At least one group mean is different")

print(f"The F test statistic is {round(f,3)} and the p-value is {round(p_val,4)}")

if p_val < alpha:
    print("Conclusion: Reject Ho: At least one group mean is different")
    ANOVAtype = "ANOVA: At least one group mean different"
else:
    print("Conclusion: Fail to Reject Ho: We can't reject that the means are the same")
    ANOVAtype = "ANOVA: Group Means are the same"

print()
print("Region by TotalSales columns")
print(AthletesDF.pivot_table(['Points'], index=['Region']))

# Box plot of means for Regions
y=[Northeast['Points'],
   South['Points'],
   Midwest['Points'],
   West['Points']]
plt.boxplot(y)
plt.title(f'F: {round(f,3)}, p-val: {round(p_val,4)}',size=10)
plt.suptitle(ANOVAtype,size=10)
plt.xticks(range(1,5), [f"Northeast: {round(Northeast['Points'].mean(),2)}",
                        f"South: {round(South['Points'].mean(),2)}", 
                        f"Midwest: {round(Midwest['Points'].mean(),2)}",
                        f"West: {round(West['Points'].mean(),2)}"])
plt.ylabel('Points')
plt.savefig('ANOVAPROJECT.png', bbox_inches='tight')
plt.show()


# Tukey pairwise comparison

from statsmodels.stats.multicomp import pairwise_tukeyhsd


# Data (endogenous/response variable)
tukey = pairwise_tukeyhsd(endog=AthletesDF['Points'],
                          groups=AthletesDF['Region'], alpha=0.2)

print(tukey.summary() )



# Plot group confidence intervals
tukey.plot_simultaneous()  # 95% confidence interal plot
plt.vlines(x=AthletesDF['Points'].mean(),ymin=-1,ymax=7, color="red")
plt.show()

Pub=AthletesDF[AthletesDF['PublicOrPrivate']=='Public']
Priv=AthletesDF[AthletesDF['PublicOrPrivate']=='Private']

alpha = .05
tvar, p_valvar = stats.bartlett(Pub['Points'],Priv['Points'])

print("This is a test of equal variances")
print("Ho: The variances are equal")
print("Ha: The variances are not equal")
print(f"The t test statistic is {round(tvar,3)} and the p-value is {round(p_valvar,4)}")
print()
print("If p-value is less than alpha, then reject Ho")
print("p-value = " +str(round(p_valvar,4)))
print ("alpha = .05")
if p_valvar < alpha :
    print("Because p-value is less than alpha")
    print("Conclusion: Reject Ho: The variances are not equal")
    tEqVar=False
    ttype='Welch (unequal variances) Two-Sample t test'
else:
    print("Because p-value is not less than alpha")
    print("Conclusion: Fail to Reject Ho: We can't reject that the variances are the same")
    tEqVar=True
    ttype='Two-Sample t test (assuming equal variances)'
    

print()
print()

print("Test for the means of 2 independent variables")
print("Ho:  The means for Points, regardless of public or private, are the same")
print("Ha:  The means for Points, regardless of public or private, are not the same")
print("Alpha = .05")
alpha = .05
tmean, p_valmean = stats.ttest_ind(Pub['Points'],
                                   Priv['Points'],equal_var=tEqVar)
print()
print()


print("This is a " + ttype + " of equal means")
print("The means for Points, regardless of public or private, are the same")
print("Ha:  The means for Points, regardless of public or private, are not the same")


print(f"The t test statistic is {round(tmean,3)} and the p-value is {round(p_valmean,4)}")


print()
print("If p-value is less than alpha, then reject Ho")
print("p-value = " +str(round(p_valmean,4)))
print ("alpha = .05")
if p_valmean < alpha:
    print("Because p-value is less than alpha")
    print("Conclusion: Reject Ho: The means are not equal")
else:
    print("Because p-value is greater than alpha")
    print("Conclusion: Fail to Reject Ho: We can't reject that the means are the same")
    
# Create the boxplot
y=[Pub['Points'],Priv['Points']]
plt.boxplot(y)
plt.title(f't: {round(tmean,3)}, p-val: {round(p_valmean,4)}', size=10)
plt.suptitle(ttype, size=10)
plt.xticks(range(1,3),[f"Public: {round(Pub['Points'].mean(),2)}",
f"Private: {round(Priv['Points'].mean(),2)}"])
plt.ylabel('Points per Game')
plt.savefig('ttest.png', bbox_inches='tight')
plt.show()

# Second 2 sample hypothis test
IvyReg=SchoolsDF[SchoolsDF['Ivy']=='Y']
NReg=SchoolsDF[SchoolsDF['Ivy']=='N']

alpha = .05
tvar, p_valvar = stats.bartlett(IvyReg['Tuition'],NReg['Tuition'])

print("This is a test of equal variances")
print("Ho: The variances are equal")
print("Ha: The variances are not equal")
print(f"The t test statistic is {round(tvar,3)} and the p-value is {round(p_valvar,4)}")
print()
print("If p-value is less than alpha, then reject Ho")
print("p-value = " +str(round(p_valvar,4)))
print ("alpha = .05")
if p_valvar < alpha :
    print("Because p-value is less than alpha")
    print("Conclusion: Reject Ho: The variances are not equal")
    tEqVar=False
    ttype='Welch (unequal variances) Two-Sample t test'
else:
    print("Because p-value is not less than alpha")
    print("Conclusion: Fail to Reject Ho: We can't reject that the variances are the same")
    tEqVar=True
    ttype='Two-Sample t test (assuming equal variances)'
    

print()
print("Test for the means of 2 independent variables")
print("Ho:  The means for Tuition, regardless of Ivy or not, are the same")
print("Ha:  The means for Tuition, regardless of Ivy or not, are not the same")
print("Alpha = .05")
alpha = .05
tmean, p_valmean = stats.ttest_ind(IvyReg['Tuition'],
                                   NReg['Tuition'],equal_var=tEqVar)
print()
print()


print("This is a " + ttype + " of equal means")
print("Ho:  The means for Tuition, regardless of Ivy or not, are the same")
print("Ha:  The means for Tuition, regardless of Ivy or not, are not the same")


print(f"The t test statistic is {round(tmean,3)} and the p-value is {round(p_valmean,4)}")


print()
print("If p-value is less than alpha, then reject Ho")
print("p-value = " +str(round(p_valmean,4)))
print ("alpha = .05")
if p_valmean < alpha:
    print("Because p-value is less than alpha")
    print("Conclusion: Reject Ho: The means are not equal")
else:
    print("Because p-value is greater than alpha")
    print("Conclusion: Fail to Reject Ho: We can't reject that the means are the same")
    
# Create the boxplot
y=[NReg['Tuition'],IvyReg['Tuition']]
plt.boxplot(y)
plt.title(f't: {round(tmean,3)}, p-val: {round(p_valmean,4)}', size=10)
plt.suptitle(ttype, size=10)
plt.xticks(range(1,3),[f"NReg: {round(NReg['Tuition'].mean(),2)}",
f"IvyReg: {round(IvyReg['Tuition'].mean(),2)}"])
plt.ylabel('Tuition')
plt.savefig('ttest.png', bbox_inches='tight')
plt.show()

SnowAccumulated = { 'Denver' : [26,48,58,80],
             'Chicago' : [40,38,34,56],
             'NewYork' : [109,90,70,78],
                }

SA = DataFrame(SnowAccumulated)
print (SA)

IndexNames = ['2017-2018', '2018-2019', '2019-2020', '2020-2021']
SA.index=IndexNames
print (SA)

wb=xw.Book()                
sht = wb.sheets['Sheet1']   

sht.range('A1').value = SA

# Add labels in appropriate positions
sht.range('A6').value = 'AVG'
sht.range('E1').value = 'AVG'

# Add average computations in appropriate places
sht.range('E2').value = SA.loc['2017-2018'].mean()
sht.range('E3').value = SA.loc['2018-2019'].mean()
sht.range('E4').value = SA.loc['2019-2020'].mean()
sht.range('E5').value = SA.loc['2020-2021'].mean()

# Add average computations in appropriate positions
sht.range('B6').value = SA.Denver.mean()
sht.range('C6').value = SA.Chicago.mean()
sht.range('D6').value = SA.NewYork.mean()

# Create and add a chart based on data in the workbook
chart = sht.charts.add (left=0, top=100, width=355, height=200)
chart.set_source_data (sht.range('A1:D5'))

chart = sht.charts.add (left=400, top=100, width=350, height=200)
chart.set_source_data (sht.range('A1:C5'))


chart = sht.charts.add (left=200, top=400, width=350, height=200)
chart.set_source_data (sht.range('A1:b5'))

AthletesDF.plot(
    kind='hexbin',
    x='Assists',
    y='Points',
    C='Height',
    gridsize=20,
    figsize=(12,8),
    cmap="Blues",
    sharex=False)
plt.show()

AthletesDF.groupby(
    ['Region']
)['NumberofGamesPlayed'].mean().plot(
    kind='pie',
    figsize=(12,8),
    cmap="Blues_r",
    explode = (0, 0.1, 0, 0))
plt.show()

plt.hist(SchoolsDF.Tuition, 6, color = "skyblue", ec="skyblue")
mn = str(round(SchoolsDF.Tuition.mean(),2))
sd = str(round(SchoolsDF.Tuition.std(),2))
plt.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.6)
plt.title(f'Histogram of Tuitions: mu= {mn}, sigma={sd}')
plt.xlabel('Tuition Price')
plt.ylabel('Frequency')
plt.show()