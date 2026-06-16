import paho.mqtt.client as mqtt
import json

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

    print(f"{timestamp}     {motion}    {obstacle}      {humidity}      {temp}      {mic_level}     {mq2_raw}")


def on_disconnect(client, userdata, flags, reason_code, properties):
    print(f"Disconnected ({reason_code})")


def main():
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
    main()