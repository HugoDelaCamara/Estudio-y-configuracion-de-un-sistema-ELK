# Importacion de librerias necesarias
import json
import random
import time
from datetime import datetime, timedelta

# Ruta al archivo donde se almacenan los datos
path = r'C:\Users\hugod\Desktop\TFG\ficheros\escenario5.log'

# Informacion que contendran los campos del mensaje
customers = [123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 139]
payment_methods = ["Credit Card", "Debit Card", "PayPal"]
product_categories = ["Electronics", "Books", "Clothing"]

# Funcion que genera registros en funcion de los campos indicados
def generate_transaction(transaction_id, timestamp):
    customer_id = random.choice(customers)
    amount = round(random.uniform(20.0, 300.0), 2)
    payment_method = random.choice(payment_methods)
    product_category = random.choice(product_categories)
    
    transaction = {
        "transaction_id": str(transaction_id),
        "timestamp": timestamp.isoformat() + "Z",
        "customer_id": str(customer_id),
        "amount": amount,
        "payment_method": payment_method,
        "product_category": product_category,
    }
    return transaction

# Bucle que genera registros hasta que se detiene el script
with open(path, 'a') as log_file:
    transaction_id = 1
    current_time = datetime.utcnow()
    
    while True:  
        transaction = generate_transaction(transaction_id, current_time)
        log_file.write(json.dumps(transaction) + '\n')
        log_file.flush() 
        
        transaction_id += 1
        current_time += timedelta(minutes=5) 

        time.sleep(1) 
