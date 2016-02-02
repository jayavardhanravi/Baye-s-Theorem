import numpy as n
import math
import matplotlib.pyplot as plt

# Declaring global variables
x = []
y = []
xmean = []
ymean = []
xstd = []
ystd = []

# Function to read the data from file
def read_file(par_filename):
	with open(par_filename,"r") as file_lines:
		features = [[float(i) for i in line.split()] for line in file_lines]
	file_lines.close()
	return features;

# Function to compute the means and the standard deviations
def cal_mean_std(par_features):
	global x
	global y
	for i in par_features:
		if(i[12]==1):
			x.append(i)
		if(i[12]==2):
			y.append(i)
	global xmean
	global ymean
	# Calculating the means of each class features		
	xmean = n.mean(x,0)
	ymean = n.mean(y,0)
	global xstd
	global ystd
	# Calculating the standard deviation of each class features
	xstd = n.std(x,0)
	ystd = n.std(y,0)
	return;
	
# Function to compute the class1 probabilities (priori probability)
def cal_class_probability(par_features,par_class):
	cx = 0
	cy = 0
	for j in par_features:
		if(j[12]==1):
			cx = cx + 1
		if(j[12]==2):
			cy = cy + 1
	if(par_class==1):
		p_c = float(cx)/float((cx+cy))
	elif(par_class==2):
		p_c = float(cy)/float((cx+cy))
	else:
		p_c = 0
	return p_c;
	
# Function to compute the probability of feature X given the hypothesis
def cal_posteriori_probability(par_feature_val,par_mean_val,par_std_val):
	x1 = 1/(math.sqrt(2*(math.pi)*(math.pow(par_std_val,2))))
	y1 = par_feature_val - par_mean_val
	x2 = math.exp((-1)*((math.pow(y1,2))/(2*math.pow(par_std_val,2))))
	p = x1*x2
	return p;
	
# Function to compute the posteriori probability of each feature
def cal_posteriori_each_feature(par_input_features,par_input_means,par_input_std):
	p_each = []
	for k in par_input_features:
		p_e = []
		for l in range(0,11): 
			p_e.append(cal_posteriori_probability(k[l],par_input_means[l],par_input_std[l]))
		p_each.append(p_e)
	return p_each;
	
# Function to compute the priori probability of features X
def cal_priori_features(par_pc1,par_pc2,par_ppc1,par_ppc2):
	ppc1 = []
	for q in par_ppc1:
		product = 1
		for w in range(0,11):
			product *= q[w]
		ppc1.append(product)
	ppc2 = []
	for e in par_ppc2:
		product1 = 1
		for r in range(0,11):
			product1 *= e[r]
		ppc2.append(product1)
	px1 = [t*par_pc1 for t in ppc1]
	px2 = [t1*par_pc2 for t1 in ppc2]
	x = list(n.array(px1)+n.array(px2))
	return x;
	
# Function to compute the probability of hypothesis given feature X
def cal_posteriori_class(par_px,par_pc,par_ppc):
	ppc = []
	for q1 in par_ppc:
		product11 = 1
		for w1 in range(0,11):
			product11 *= q1[w1]
		ppc.append(product11)
	qx = list(n.array(ppc)/n.array(par_px))
	qx1 = [par_pc*q2 for q2 in qx]
	return qx1;
	
# Function to create a ground truth labels
def create_labels(par_features):
	lc = []
	for i1 in par_features:
		lc.append(i1[12])
	print "lc data : ",lc
	return lc;
	
# Function to create the expected labels
def create_expected_labels(par_p1,par_p2):
	elc = []
	if(len(par_p1)==len(par_p2)):
		for a1 in range(0,len(par_p1)):
			if(par_p1[a1]>par_p2[a1]):
				elc.append(1)
			elif(par_p1[a1]<par_p2[a1]):
				elc.append(2)
			else:
				elc.append(0)
	return elc;
	
# Function to compare labels to compute training error
def cal_training_error(par_truth,par_gen):
	matched = 0
	mismatched = 0
	print "count truth : ",len(par_truth)
	print "count gen : ",len(par_gen)
	if(len(par_truth)==len(par_gen)):
		for m in range(0,len(par_truth)):
			if(par_truth[m]==par_gen[m]):
				matched = matched + 1
			if(par_truth[m]!=par_gen[m]):
				mismatched = mismatched + 1
	error = ((float(mismatched))/(float(matched+mismatched)))*100	
	print "Number of mismatches : ", mismatched
	return error;
	
# Function to compute the confucion matrix
def cal_confusion_matrix(par_truth1,par_gen1):
	ma1 = 0
	ma2 = 0
	misma1 = 0
	misma2 = 0
	for zs in range(0,len(par_truth1)):
		if(par_truth1[zs]==1):
			ma1 = ma1+1
		if(par_truth1[zs]==2):
			ma2 = ma2+1
	for zs1 in range(0,len(par_gen1)):
		if(par_gen1[zs1]==1):
			misma1 = misma1+1
		if(par_gen1[zs1]==2):
			misma2 = misma2+1
	print "Predicted Class 1 :",ma1
	print "Predicted Class 2 :",ma2
	print "Actual Class 1 :",misma1
	print "Actual Class 2 :",misma2
	return;
			
# Function to clear the global variables
def clear_variables():
	del x[:]
	del y[:]
	global xmean
	xmean = []
	global ymean
	ymean = []
	global xstd
	xstd = []
	global ystd
	ystd = []
	return;
	
# Function to compute error for training set
def cal_each_trainingset(xt):
	clear_variables();
	data_from_file = xt
	cal_mean_std(data_from_file);
	print "-------------------------------Training-----------------------------------"
	p_c1 = cal_class_probability(data_from_file,1);
	p_c2 = cal_class_probability(data_from_file,2);
	post_prob_c1 = cal_posteriori_each_feature(data_from_file,xmean,xstd);
	post_prob_c2 = cal_posteriori_each_feature(data_from_file,ymean,ystd);
	priori_x = cal_priori_features(p_c1,p_c2,post_prob_c1,post_prob_c2);
	post_final_c1 = cal_posteriori_class(priori_x,p_c1,post_prob_c1);
	post_final_c2 = cal_posteriori_class(priori_x,p_c2,post_prob_c2);
	truth_labels = create_labels(data_from_file);
	gen_labels = create_expected_labels(post_final_c1,post_final_c2);
	training_error = cal_training_error(truth_labels,gen_labels);
	cal_confusion_matrix(truth_labels,gen_labels);
	print "Trainig Error is : ",training_error
	return training_error;
	
# Function to compute error for testing set
def cal_each_testingset(xtes,ytes):
	clear_variables();
	cal_mean_std(xtes);
	print "-------------------------------Testing-----------------------------------"
	p_c1 = cal_class_probability(xtes,1);
	p_c2 = cal_class_probability(xtes,2);
	post_prob_c1 = cal_posteriori_each_feature(ytes,xmean,xstd);
	post_prob_c2 = cal_posteriori_each_feature(ytes,ymean,ystd);
	priori_x = cal_priori_features(p_c1,p_c2,post_prob_c1,post_prob_c2);
	post_final_c1 = cal_posteriori_class(priori_x,p_c1,post_prob_c1);
	post_final_c2 = cal_posteriori_class(priori_x,p_c2,post_prob_c2);
	truth_labels = create_labels(ytes);
	#print " before count truth : ",len(truth_labels)
	gen_labels = create_expected_labels(post_final_c1,post_final_c2);
	#print " befre count gen : ",len(gen_labels)
	testing_error = cal_training_error(truth_labels,gen_labels);
	#print "Testing Error is : ",testing_error
	cal_confusion_matrix(truth_labels,gen_labels);
	return testing_error;
	
# Function to split the data based on percentage
def split_data(f,percent):
	tot = len(f)
	req_xt = int((float(percent)/100)*(tot))
	req_yt = tot - req_xt
	xt_get = []
	for s in range(0,(req_xt-1)):
		xt_get.append(f[s])
	yt_get = []
	for d in range(req_xt,tot):
		yt_get.append(f[d])
	xyt = []
	xyt.append(xt_get)
	xyt.append(yt_get)
	return xyt;
	
# Function to compute the percentage
def cal_percent(fvar,pervar):
	totvar = len(fvar)
	req_xvar = int((float(pervar)/100)*(totvar))
	req_yvar = totvar - req_xvar
	obt_percent = ((float(req_xvar)/totvar))*100
	return obt_percent;
	
# Function to compute
def compute_func(f1,perc):
	xyt1 = split_data(f1,perc);
	x111 = xyt1[0]
	y111 = xyt1[1]
	print "Length of x111 : ",len(x111)
	print "Length of y111 : ",len(y111)
	train_error = cal_each_trainingset(x111);
	test_error = cal_each_testingset(x111,y111);
	err_ret = []
	err_ret.append(train_error)
	err_ret.append(test_error)
	return err_ret;
	
# Starting of the flow of the program
data_from_file = read_file("plrx.txt");
input_percent = [40, 50, 60, 70, 80, 90]
file_created = open('Generated_error_table.dat','w')
for pri in range(0,len(input_percent)):
	print "###########Computing for #############",input_percent[pri]
	yui = compute_func(data_from_file,input_percent[pri])
	per_obt = cal_percent(data_from_file,input_percent[pri])
	file_created.write("%f %f %f \n" %(yui[0],yui[1],per_obt))
file_created.close()

test_plt = []
train_plt = []
percent_plt = []
with open("Generated_error_table.dat","r") as lines_in_file:
	for c1 in lines_in_file:
		test_plt.append(c1.split()[0])
		train_plt.append(c1.split()[1])
		percent_plt.append(c1.split()[2])
		
fig = plt.figure()
ax = fig.add_subplot(111)
		
plt.plot(percent_plt, test_plt, 'bo', label='Training Error')
plt.plot(percent_plt, train_plt, 'ro', label='Testing Error')
plt.plot(percent_plt, test_plt, 'b')
plt.plot(percent_plt, train_plt, 'r')

ax.set_xlabel('Percentage of Taining data')
ax.set_ylabel('Percentage of Error')
plt.legend( loc='upper left', numpoints = 1 )
plt.title("% Error Vs % training Data")

plt.show()
		
			
