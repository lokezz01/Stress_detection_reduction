#import flask
from flask import Flask,redirect,url_for,render_template,request
import pandas
import sys
import ui

app = Flask(__name__)

score = 0

"""@app.route('/', methods=['GET'])
def return_data():
    return render_template("login.html")"""

@app.route("/login", methods=["POST","GET"])

def login():
    global score
    if request.method == "POST":
        val = request.form
        if val["n1"] == "YES":
            score += 5
        else:
            score += 3

        if val["n3"] == "1 - 3 hours":
            score += 3

        else:
             score += 5
        
        if val["n5"] == "Less than a week":
            score += 3

        else:
            score += 5

        if val["n7"] == "Less than a week":
            score += 3

        else:
            score += 5

        if val["n9"] == "Below 30%":
            score += 1

        elif val["n9"] == "Below 60%":
            score += 3

        else:
            score += 5

        if val["n12"] == "YES":
            score += 5

        else:
            score += 3

        if val["n14"] == "YES":
            score += 5

        else:
            score += 3

        if val["n16"] == "YES":
            score += 5

        else:
            score += 3

        #changed_eyebrow.get_data(score)
        
        ui.final_stress(score)
        
        #return str(score)
        
        return redirect(request.url)
        
        #print("yes ",score)
        
      
    else:
        score = 0
        return render_template("login.html")

def main():
    score = 0
    app.run()


if __name__ == "__main__":
    #changed_eyebrow.main()
    main()
    
