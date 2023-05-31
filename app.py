from flask import Flask,render_template,request,jsonify
#from chat import get_response
from summarizer_bot import chat,summarizer,summary_api,chatt5
from flask_cors import CORS
import json

app=Flask(__name__)


response=''
@app.get('/')
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    j=request.get_json()#input from user
    text=j.get("message")
    #todo: check if text is valid
    if len(text)>100:
        response=chatt5(text)
    elif text.startswith("https"):
        response=summary_api(text)
    else:
        response=chat(text)
        response=str(response)
        clean_text = response.replace('<s>', "").replace('[','').replace(']','').replace("'","").replace("'","")
        response=clean_text
        #clean_text=str(clean_text)
    #response=get_response(text)
    message={"answer":response}
    return jsonify(message)#output giving to js 



if __name__=="__main__":
    app.run(debug=True)
    
