import pandas as pd
import numpy as np

data = pd.read_csv("error_analysis_inceptionV3_masked_sgd_25e4lr_1e4dc_9e1f_30e_64b_.csv")
# data = pd.read_csv("error_analysis_inceptionV3_masked_sgd_25e4lr_1e4dc_9e1f_30e_64b_.csv")
categories = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
v_true = data['valid_true']
v_pred = data['valid_pred']
333
result = dict()
error = dict()
for cat in categories:
	true, wrong = 0, 0
	for i in range(len(data)):
		if v_true[i] == cat:
			if v_pred[i] == v_true[i]:
				true +=1
			else:
				wrong +=1

	result[cat] = {"Positive": true, "Negative": wrong}

	
for cat in categories:
	print (cat+": ", result[cat], "-- {0:.2f}%".format((result[cat]["Positive"] / (result[cat]["Positive"] + result[cat]["Negative"])*100 )))

for a in categories:
	tmp = dict()
	for b in categories:
		ccc = 0	
		for i in range(len(data)):
			if v_true[i] == a:
				if v_pred[i] == b:
					ccc += 1

		tmp[b] = ccc
	error[a] = tmp


for cat in categories:
	print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	print (cat)
	print (error.get(cat))

	