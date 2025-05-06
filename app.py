from flask import Flask
from flask_cors import CORS
from routes.lecture_routes import router

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000","localhost:7000"])
app.register_blueprint(router)
if __name__ == '__main__':
    app.run(debug=True)