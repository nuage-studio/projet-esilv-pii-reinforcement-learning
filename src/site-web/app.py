from flask import Flask ,request
from flask import render_template
import pandas as pd
import joblib
from io import BytesIO
import base64
import shap
from keras.models import load_model
import numpy as np
import glob


data=pd.read_csv("churn_telec.csv")
data = data[data["Churn"] == "Yes"]

loaded_model = joblib.load("decisiontree")
deep_model=load_model("model.h5")


x_df,x_train=pd.read_csv("x_df.csv"),pd.read_csv("x_train.csv")
x_df_deep,x_train_deep=pd.read_csv("x_df_deep.csv"),pd.read_csv("x_train_deep.csv")


x_train=x_train.drop("customerID",axis=1)
x_train_deep=x_train_deep.drop("customerID",axis=1)


columns=data.columns
values=data.values




explainer = shap.KernelExplainer(loaded_model.predict,x_train)
explainer_deep = shap.KernelExplainer(deep_model.predict,x_train_deep)





def find_file_deep(code):
	files=glob.glob("./Templates/deep_learning/*.html", recursive = True)
	return ("./Templates/deep_learning\\"+code+"deep.html") in files

def find_file(code):
	files=glob.glob("./Templates/machine_learning/*.html", recursive = True)
	return ("./Templates/machine_learning\\"+code+".html") in files





def blackbox(explainer, row):
    if(find_file(row.customerID.values[0])):
    	return (row.customerID.values[0]+".html")
    else:
	    shap_values = explainer.shap_values(row.drop("customerID",axis=1),nsamples=100)
	    shap.initjs()
	    f=shap.force_plot(explainer.expected_value, shap_values, row.drop("customerID",axis=1))
	    shap.save_html("./Templates/machine_learning/"+row.customerID.values[0]+".html", f)
	    return (row.customerID.values[0]+".html")



def blackbox_deep(explainer, row,code):
	if(find_file_deep(code)):
		return (code+"deep.html")
	else:
	    shap_values = explainer.shap_values(row,nsamples=100)
	    shap.initjs()
	    f=shap.force_plot(explainer.expected_value, shap_values[0], row)
	    shap.save_html("./Templates/deep_learning/"+code+"deep.html", f)
	    return (code+"deep.html")


app=Flask(__name__)





@app.route('/')
def index():
	return render_template('home.html',file_ds="explainability_global.html")




@app.route('/user',methods=["GET" , "POST"])
def function():
	id=request.args.get('id')
	row=(pd.DataFrame(x_df[x_df["customerID"]==id].iloc[-1]).T)
	pred=loaded_model.predict(row.drop("customerID",axis=1))[0]
	file=blackbox(explainer,row)

	row_deep=(pd.DataFrame(x_df_deep[x_df_deep["customerID"]==id].iloc[-1]).T)
	n=np.asarray(row_deep.drop("customerID",axis=1)).astype(np.float32)
	pred_deep=deep_model.predict(n)	
	file_deep=blackbox_deep(explainer_deep,n,row_deep.customerID.values[0])

	return render_template('index.html',id=id,pred=pred,file="./machine_learning/"+file,pred_deep=pred_deep[0][0],file_deep="./deep_learning/"+file_deep,tenure=data[data["customerID"]==id].tenure.values[0])





@app.route('/clients')
def functions():
	return render_template('clients.html',columns=columns,data=values,lencolumns=len(columns))


if __name__ == "__main__":
	app.run(host="127.0.0.1", port=5000, debug=True)