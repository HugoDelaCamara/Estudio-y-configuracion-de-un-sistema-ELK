import requests
import json
import websocket

def rename_fields(message):
    data = json.loads(message)
    if "data" in data:
        renamed_data = []
        for item in data["data"]:
            renamed_item = {
                "Symbol": item.get("s"),
                "LastPrice": item.get("p"),
                "Timestamp": item.get("t"),
                "Volume": item.get("v"),
                "TradeConditions": item.get("c")
            }
            renamed_data.append(renamed_item)
        data["data"] = renamed_data
    return json.dumps(data)

def on_message(ws, message):
    url = 'http://localhost:8080'
    headers = {'Content-Type': 'application/json'}
    modified_message = rename_fields(message)
    response = requests.post(url, headers=headers, data=modified_message)
    print(response.text)

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
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
