from Calculations import Calculate

# change the arguments of x and run this script to get the results.
# default values are set for columnname and correlation name. If you want other columns or are using a
# different file, specify them.
# x = c(ddofnumber, 'pathe\csvfile.csv', 'columnname(optional', 'column to correlate with(optional')

x = Calculate(1, 'tomslee_airbnb_amsterdam_1476_2017-07-22.csv')
x.runall()
