# -*- coding: utf-8 -*-

"""
Written by Kambana on April 21, 2026
"""

import pika
import subprocess
import json
import time
import re

RABBITMQ_HOST = "localhost"
QUEUE_NAME = "zeek_conn_queue"
INTERFACE = "ens6"

# Robust RabbitMQ connection
def create_connection():
    while True:
        try:
            parameters = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                heartbeat=600,               # Heartbeat every 10 min
                blocked_connection_timeout=300
            )
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            
            # Declare a non-persistent queue (durable=False)
            channel.queue_declare(queue=QUEUE_NAME, durable=False)
            
            return connection, channel
        except Exception as e:
            print("RabbitMQ connection error, retrying in 1s:", e)
            time.sleep(1)

# Clean JSON / UTF-8

def fix_invalid_utf8(line: str) -> str:
    # Properly decode \xNN sequences into UTF-8 characters
    try:
        line = bytes(line, "utf-8").decode("unicode_escape")
    except Exception:
        pass

    # Remove all non-printable invalid characters
    line = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', line)

    return line


# Fire up Zeek
zeek_cmd = [
    "/opt/zeek/bin/zeek",
    "-i", INTERFACE,
    "-C",
    "-f", "port 4840 or port 1880 or port 52380 or port 4444", #Avoid spirus variable from other network flow for experiment
    "LogAscii::use_json=T",
    "/opt/zeek/share/zeek/site/local.zeek"
]

proc = subprocess.Popen(
    zeek_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

print("Zeek started, sending events to RabbitMQ...")

# Initial connection
connection, channel = create_connection()


# Main loop for reading and publishing

for line in proc.stdout:
    line = line.strip()
    if not line:
        continue

    try:
        clean_line = fix_invalid_utf8(line)
        record = json.loads(clean_line)

        # Publish with automatic reconnection
        while True:
            try:
                channel.basic_publish(
                    exchange="",
                    routing_key=QUEUE_NAME,
                    body=json.dumps(record),
                    properties=pika.BasicProperties(delivery_mode=1)  # non-persistent
                )
                break  # success
            except pika.exceptions.AMQPError as e:
                print("Publish error, reconnecting:", e)
                try:
                    connection.close()
                except Exception:
                    pass
                connection, channel = create_connection()

    except json.JSONDecodeError as e:
        print("JSON error:", e)
        print("Raw line:", line)
