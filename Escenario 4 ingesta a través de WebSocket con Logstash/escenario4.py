# Importacion de librerias necesarias
import requests
import json
import websocket

# Funcion que renombra los campos mandados en el mensaje
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

# Funcion que define el contenido del mensaje y a donde va a ser mandado
def on_message(ws, message):
    url = 'http://localhost:8080'
    headers = {'Content-Type': 'application/json'}
    modified_message = rename_fields(message)
    response = requests.post(url, headers=headers, data=modified_message)
    print(response.text)

# Funcion que indica si ocurre algun error en el procesamiento
def on_error(ws, error):
    print(error)

# Funcion que indica el cierre del procesamiento
def on_close(ws):
    print("### closed ###")

# Funcion que indica a que divisas se suscribe
def on_open(ws):
    ws.send('{"type":"subscribe","symbol":"BINANCE:ETHUSDT"}')
    ws.send('{"type":"subscribe","symbol":"BINANCE:XRPUSDT"}')
    ws.send('{"type":"subscribe","symbol":"BINANCE:BTCUSDT"}')
    ws.send('{"type":"subscribe","symbol":"BINANCE:LTCUSDT"}')
    ws.send('{"type":"subscribe","symbol":"BINANCE:BCHUSDT"}')

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("wss://ws.finnhub.io?token=cp0dqk9r01qnigejuh40cp0dqk9r01qnigejuh4g",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()
