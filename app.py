import datetime
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import pickle
import plotly.express as px
import plotly
import json
import numpy as np
from numpy import dtype
import pandas as pd
from flask import (
    Flask,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for
)
import itertools

class User:
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

    def __repr__(self):
        return f'<User: {self.username}>'



users = []
users.append(User(id=1, username='team2', password='mock'))

app = Flask(__name__)
app.secret_key = 'team2mock'

@app.before_request
def before_request():
    g.user = None

    if 'user_id' in session:
        user = [x for x in users if x.id == session['user_id']][0]
        g.user = user
        

@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'team2' or request.form['password'] != 'mock':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('profile'))
    return render_template('login.html', error=error)

@app.route('/profile', methods=['GET'])
def profile():
    if not g.user:
        return redirect(url_for('login'))
    
    
     
    
    return render_template('prof.html')

@app.route('/result', methods=['POST'])
def result():

#get date from html    
    full_date = request.form.get("try")
    
    fmd = pd.to_datetime(full_date[4:15])

    ld = pd.to_datetime(full_date[18:])
#request model from html
    model = request.form.get("model") 
#test data and ytest & import    
    test = pd.read_parquet(r'Treat_test_final2.parquet')
    y_act = pd.read_csv(r'y_test.csv')
#change dtype of datetime    
    test['datetime'] = pd.to_datetime(test['datetime'])
    y_act['datetime'] = pd.to_datetime(test['datetime'])
#filter_date
    filter_date = test[(test.datetime >= fmd ) &( test.datetime <= ld)]
    filter_actualdate = y_act[(y_act.datetime >= fmd ) &( y_act.datetime <= ld)]
#to predict we dont need id and date    
    x_test1=filter_date.drop(['tracking_id','datetime'],axis=1)
    pred_model = pd.DataFrame()

#to predict we dont need id and date    
    y_act1=filter_actualdate.drop(['tracking_id','datetime'],axis=1)
    pred_actual = pd.DataFrame(y_act1)

#----------xg_model---------------
    if(model == 'XGB Regressor'):
        result1 = 'XGB_Regressor'
        
        loaded_model = pickle.load(open('xgb_model.sav', 'rb'))
        pred_y=loaded_model.predict(x_test1)
        pred_model = pd.concat([pred_model,pd.DataFrame(pred_y)])
        
        
        mod_def = 'Gradient boosting refers to a class of ensemble machine learning algorithms that can be used for classification or regression predictive modeling problems.'

#----------extra_tree_model---------------
    
    elif(model == 'Extra Tree Regressor'):
        result1 = 'Extra_Tree_Regressor'
        
        loaded_model = pickle.load(open('extra_tree_model.sav', 'rb'))
        pred_y=loaded_model.predict(x_test1)
        pred_model = pd.concat([pred_model,pd.DataFrame(pred_y)])
        
        
        mod_def = 'An extra-trees regressor. This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.'

#----------Random_forest_model---------------
    
    elif(model == 'Random Forest'):
        result1 = 'Random_Forest'
       
        loaded_model = pickle.load(open('rf_model.sav', 'rb'))
        pred_y=loaded_model.predict(x_test1)
        pred_model = pd.concat([pred_model,pd.DataFrame(pred_y)])
        
        mod_def = 'A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting'
        
#----------KNN_model---------------

    elif(model == 'KNN Regression'):
        result1 = 'KNN_Regression'#load  model and predict # graph in grid2
        
        loaded_model = pickle.load(open('KNN_model.sav', 'rb'))
        pred_y=loaded_model.predict(x_test1)
        pred_model = pd.concat([pred_model,pd.DataFrame(pred_y)])
        
        mod_def = 'KNN regression is a non-parametric method that, in an intuitive manner, approximates the association between independent variables and the continuous outcome by averaging the observations in the same neighbourhood.'
#invalid    
    else:
        result1 = 'INVALID CHOICE'
    entry = result1
    
    final = pd.concat([pred_model.reset_index(drop=True),pred_actual.reset_index(drop=True)],axis=1)
    final.columns =['model_val','actual_val']
    #read test sheet filter to the range; 

    mae= mean_absolute_error(pred_model,pred_actual)
    
    mse = mean_squared_error(pred_model,pred_actual)
    
    rmse = np.sqrt(mse)
    r2 = r2_score(pred_model,pred_actual)
    


    

    fig = px.line(final, y = ['actual_val', 'model_val'])
    graphs = [fig]
    # Add "ids" to each of the graphs to pass up to the client
    # for templating
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]

    # Convert the figures to JSON
    # PlotlyJSONEncoder appropriately converts pandas, datetime, etc
    # objects to their JSON equivalents
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)



    return render_template('prof.html', entry=entry, mod_def = mod_def, ld=ld,fmd =fmd, ids=ids,graphJSON=graphJSON, mae=mae,rmse=rmse,r2=r2)
     
    
  
                                                    

if __name__ == '__main__':
   app.run(debug = True)

