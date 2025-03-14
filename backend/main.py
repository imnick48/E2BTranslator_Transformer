from flask import Flask, request, jsonify
from os import environ as env
from db import SessionLocal
from models import Translations
from db import engine, Base
from tokenizer import Tokenizer
from translate import translate
from flask_cors import CORS
app=Flask(__name__)
CORS(app)
Base.metadata.create_all(bind=engine)
def create_entity(english,bengali):
    db=SessionLocal()
    try:
        new_entity=Translations(english=english,bengali=bengali)
        db.add(new_entity)
        db.commit()
        db.refresh(new_entity)
    except Exception as e:
        db.rollback()
        print("Error:", e)
    finally:
        db.close()
@app.route("/translate", methods=["POST"])
def translate_route():
    data = request.get_json()
    english = data.get("english")
    if not english:
        return jsonify({"error": "English text is required"}), 400
    bengali = translate(english)
    create_entity(english, bengali)
    return jsonify({"english": english, "bengali": bengali}), 201

@app.route("/translatedata",methods=["GET"])
def translate_get():
    if request.method=="GET":
        db=SessionLocal()
        data=db.query(Translations).all()
        data_output=[]
        for i in data:
            data_output.append({
                "id":i.id,
                "english":i.english,
                "bengali":i.bengali,
            })
        db.close()
    return data_output
if __name__ == "__main__":
    app.run(debug=True)