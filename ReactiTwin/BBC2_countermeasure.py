import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="

import json
import pika
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os

from datetime import datetime
from sklearn.feature_extraction import FeatureHasher

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)
tf.config.set_visible_devices([], "GPU")

RABBITMQ_HOST = "localhost"
QUEUE_NAME = "zeek_conn_queue"

MODEL_DIR = "/.../"
THRESHOLD = 0.50

ETIQUETTE = None
RESET_QUEUE_REQUESTED = False
DATE_OF_RECEPTION_MALVERSATING_FLOW = 0
DATE_OF_DETECTION = 0
DATE_OF_REMEDIATION = 0

embedder = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "embedder.h5"),
    compile=False
)

relation = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "relation.h5"),
    compile=False
)

support = tf.convert_to_tensor(
    np.load(os.path.join(MODEL_DIR, "support.npy")),
    dtype=tf.float32
)

support_labels = tf.convert_to_tensor(
    np.load(os.path.join(MODEL_DIR, "support_labels.npy")),
    dtype=tf.float32
)

ONEHOT_ENCODERS = {
    "weird_name": joblib.load(os.path.join(MODEL_DIR, "onehot_weird_name.pkl")),
    "weird_notice": joblib.load(os.path.join(MODEL_DIR, "onehot_weird_notice.pkl")),
    "weird_addl": joblib.load(os.path.join(MODEL_DIR, "onehot_weird_addl.pkl")),
    "ssl_resumed": joblib.load(os.path.join(MODEL_DIR, "onehot_ssl_resumed.pkl")),
    "ssl_established": joblib.load(os.path.join(MODEL_DIR, "onehot_ssl_established.pkl")),
}

SCALER = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

def encode_proto(proto):
    return {"TCP": 1, "UDP": 0, "ICMP": 2}.get(proto, 3)

def preprocess_single_row(row: dict) -> np.ndarray:
    df = pd.DataFrame([row])

    for col in df.columns:
        if df[col].dtype == object:
            v = df.at[0, col]
            if isinstance(v, str) and v.strip().lower() in {"", "-", "nan", "null"}:
                df.at[0, col] = None

    df.replace("0.0.0.0", 0, inplace=True)

    DROP_COLS = {
        'ts','src_ip','dst_ip','type','label',
        'http_trans_depth','http_method','http_uri','http_referrer','http_version',
        'http_request_body_len','http_response_body_len','http_status_code',
        'http_user_agent','http_orig_mime_types','http_resp_mime_types',
        'dns_query','dns_qclass','dns_qtype','dns_rcode',
        'dns_AA','dns_RD','dns_RA','dns_rejected',
        'ssl_subject','ssl_issuer'
    }

    df.drop(columns=[c for c in df.columns if c in DROP_COLS], inplace=True)

    if "proto" in df.columns:
        df["proto"] = encode_proto(df.at[0, "proto"])

    for col, encoder in ONEHOT_ENCODERS.items():
        if col in df.columns:
            encoded = encoder.transform(df[[col]])
            df = pd.concat([
                df.drop(columns=[col]),
                pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
            ], axis=1)

    hasher = FeatureHasher(n_features=32, input_type="string")
    for col in ['service', 'conn_state', 'ssl_version', 'ssl_cipher']:
        if col in df.columns:
            hashed = hasher.transform([[str(df.at[0, col])]]).toarray()
            df = pd.concat([
                df.drop(columns=[col]),
                pd.DataFrame(hashed, columns=[f"{col}_hash_{i}" for i in range(32)])
            ], axis=1)

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return SCALER.transform(df).astype(np.float32)

@tf.function
def predict_one(support, query):
    supp_emb = embedder(support, training=False)
    query_emb = embedder(query, training=False)

    supp_exp = supp_emb[None, :, :]
    query_exp = query_emb[:, None, :]

    diff = tf.abs(supp_exp - query_exp)
    prod = supp_exp * query_exp

    pair = tf.concat([diff, prod], axis=-1)

    relations = relation(pair, training=False)
    relations = tf.squeeze(relations, axis=-1)

    class_scores = tf.reduce_mean(relations, axis=1)
    probs = tf.sigmoid(class_scores)

    return probs

def callback(ch, method, properties, body):
    global ETIQUETTE, RESET_QUEUE_REQUESTED, DATE_OF_RECEPTION_MALVERSATING_FLOW, DATE_OF_DETECTION, DATE_OF_REMEDIATION

    try:
        #Automatic reset of reception's date of potential malversating flow
        DATE_OF_RECEPTION_MALVERSATING_FLOW = 0
        msg = json.loads(body.decode("utf-8"))
        X = preprocess_single_row(msg)
        DATE_OF_RECEPTION_MALVERSATING_FLOW = time.time()
        #Catch reception's date of malversating flow
        print(f"[CATCH TIME] {DATE_OF_RECEPTION_MALVERSATING_FLOW}")

        probs = predict_one(
            support,
            tf.convert_to_tensor(X.reshape(1, -1))
        ).numpy()[0]

        label = int(probs > THRESHOLD)
        print(f"[PREDICTION] Probs={probs:.6f} | label={label}")

        if label == 1:
            DATE_OF_DETECTION = time.time()
            print(DATE_OF_DETECTION)

            try:
                cmd_get = (
                    "microk8s kubectl get pods "
                    "-o jsonpath='{range .items[*]}{.metadata.name} "
                    "{.metadata.labels.app} "
                    "{.status.containerStatuses[0].ready}{\"\\n\"}{end}' "
                    "| grep '^dt-' | awk '$3==\"true\"{print $2; exit}'"
                )

                result = subprocess.run(
                    cmd_get,
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                ETIQUETTE = result.stdout.strip()
                print(f"[K8S] ETIQUETTE detected: {ETIQUETTE}")

                if ETIQUETTE:
                    mitigate = subprocess.run(
                        f"microk8s kubectl delete pod -l app={ETIQUETTE} --grace-period=0 --force",
                        shell=True,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    if mitigate.returncode == 0:
                        DATE_OF_REMEDIATION = time.time()
                        print(DATE_OF_REMEDIATION)
                    else:
                        DATE_OF_REMEDIATION = "UNKNOWN"
                        print(DATE_OF_REMEDIATION)

                    RESET_QUEUE_REQUESTED = True

                    ch.basic_ack(delivery_tag=method.delivery_tag)
                    ch.stop_consuming()
                    return

            except subprocess.CalledProcessError as e:
                print("[K8S ERROR]", e.stderr)

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print("Processing error:", e)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

while True:
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST)
    )
    channel = connection.channel()

    channel.queue_declare(queue=QUEUE_NAME, durable=False)
    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=QUEUE_NAME,
        on_message_callback=callback,
        auto_ack=False
    )

    print("Waiting for RabbitMQ messages...")
    channel.start_consuming()

    if RESET_QUEUE_REQUESTED:
        print("[INFO] Deleting queue...")

        subprocess.run(
            ["sudo", "-u", "rabbitmq", "rabbitmqctl", "delete_queue", QUEUE_NAME],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print("[INFO] Recreating queue...")

        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST)
        )
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_NAME, durable=False)

        #Logging
        now = datetime.now()
        filename = now.strftime("%d_%m_%Y_%H_%M.log")

        log_dir = "/opt/zeek/share/zeek/site/logs/"
        log_path = os.path.join(log_dir, filename)

        os.makedirs(log_dir, exist_ok=True)

        with open(log_path, "w") as log_file:
            log_file.write(f"Logging of ReactiTwin Relation Network BBC2 Detection on {now.strftime("%d_%m_%Y_%H_%M")}\n")
            log_file.write(f"DATE_OF_RECEPTION_MALVERSATING_FLOW={DATE_OF_RECEPTION_MALVERSATING_FLOW}\n")
            log_file.write(f"DATE_OF_DETECTION={DATE_OF_DETECTION}\n")
            log_file.write(f"DATE_OF_REMEDIATION={DATE_OF_REMEDIATION}\n")

        RESET_QUEUE_REQUESTED = False

        try:
            connection.close()
        except:
            pass

        time.sleep(1)

        print("[INFO] Restarting consumer...")
        continue

    else:
        break
