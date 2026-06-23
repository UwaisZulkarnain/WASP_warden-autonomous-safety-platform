import paho.mqtt.client as mqtt
import json
import joblib
import numpy as np

BROKER_HOST = "localhost" #recommended localhost if same device
BROKER_PORT = 1883 #depends on running device port
TOPIC       = "sensors/#"
USERNAME    = ""
PASSWORD    = ""

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"Connected to broker at {BROKER_HOST}:{BROKER_PORT}")
        client.subscribe(TOPIC)
    else:
        print(f"Connection failed, reason code: {reason_code}")


def on_message(client, userdata, msg):
    topic   = msg.topic
    payload = msg.payload.decode("utf-8")

    data = json.loads(payload) #timestamp, motion, obstacle, humitdity, temperature, mic_level, mq2_raw
    timestamp = data['timestamp']
    motion = data['motion']
    obstacle = data['obstacle']
    humidity = data['humidity']
    temp = data['temperature']
    mic_level = data['mic_level']
    mq2_raw = data['mq2_raw']

    temperature_to_predict = temp - 6.0
    humidity_to_predict    = humidity + 29
    mic_level_to_predict   = mic_level
    mq2_raw_to_predict     = mq2_raw

    print(f"{temperature_to_predict}   {humidity_to_predict}   {mic_level_to_predict}     {mq2_raw_to_predict}")

    #PREDICT--------------------------------------
    w_env    = 0.4
    w_analog = 0.6

    test_env    = scaler_env.transform([[temperature_to_predict, humidity_to_predict]])
    test_analog = scaler_analog.transform([[mic_level_to_predict, mq2_raw_to_predict]])

    s_env    = model_env.score_samples(test_env)[0]
    s_analog = model_analog.score_samples(test_analog)[0]
    s_comb   = w_env * s_env + w_analog * s_analog

    label = "ANOMALY" if s_comb < threshold else "Normal"

    print(f"Result         : {label}")
    print(f"Combined score : {s_comb:.4f}  (threshold = {threshold:.4f})")
    print(f"Env score      : {s_env:.4f}   (temp + humidity)")
    print(f"Analog score   : {s_analog:.4f}   (mic + mq2)")
    #END PREDICT ----------------------------------------



def on_disconnect(client, userdata, flags, reason_code, properties):
    print(f"Disconnected ({reason_code})")


def mqtt_recieve():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    if USERNAME:
        client.username_pw_set(USERNAME, PASSWORD)

    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect

    print(f"Connecting to {BROKER_HOST}:{BROKER_PORT} ...")
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        client.disconnect()


if __name__ == "__main__":
    #LOAD MODEL -------------------------------------------
    model_env     = joblib.load('model/model_env.pkl')
    model_analog  = joblib.load('model/model_analog.pkl')
    scaler_env    = joblib.load('model/scaler_env.pkl')
    scaler_analog = joblib.load('model/scaler_analog.pkl')
    threshold     = float(np.load('model/threshold.npy'))

    print("✅ All models and scalers loaded")
    print(f"   Threshold : {threshold:.4f}")
    #END LOAD MODEL --------------------------------------
    mqtt_recieve()