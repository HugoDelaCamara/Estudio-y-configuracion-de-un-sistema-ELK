# API key: conrq51r01qm6hd153ogconrq51r01qm6hd153p0

#https://pypi.org/project/websocket_client/
import websocket
from elasticsearch import Elasticsearch
import datetime
import json

client = Elasticsearch("http://localhost:9200")

client.info()

def on_message(ws, message):
    message_json = json.loads(message)
    message_json["@timestamp"] = datetime.datetime.utcnow().isoformat() 
    resp = client.index(index="websockets-data", body=message_json)
    print(message_json)

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(ws):
    ws.send('{"type":"subscribe","symbol":"AAPL"}')
    ws.send('{"type":"subscribe","symbol":"AMZN"}')
    ws.send('{"type":"subscribe","symbol":"BINANCE:BTCUSDT"}')
    ws.send('{"type":"subscribe","symbol":"IC MARKETS:1"}')

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("wss://ws.finnhub.io?token=conrq51r01qm6hd153ogconrq51r01qm6hd153p0",
                              on_message = on_message,
                              on_error = on_error,
                              on_close = on_close)
    ws.on_open = on_open
    ws.run_forever()
