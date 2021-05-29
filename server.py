from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask import jsonify
import stock_app as stap
app = Flask(__name__)
api = Api(app)

class ShareMarket(Resource):
    def get(self,isin):
        # return {"hey":"sdf"}
        temp = stap.predictPrice(isin)
        return temp
class Landing(Resource):
    def get(self):
        return {"Hello":"World"}

api.add_resource(ShareMarket, '/getPrice/<isin>') # Route_1
api.add_resource(Landing, '/') # Route_1

if __name__ == '__main__':
     app.run(port='8000')