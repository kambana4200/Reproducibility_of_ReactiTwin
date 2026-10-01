#Created on We March 4 14:56:16 2026
#@author: kambana

@load base/protocols/conn
@load base/protocols/ssl

module RabbitMQ;

export {
    type ConnInfo: record {
        src_port: string &log;
        dst_port: string &log;
        proto: string &log;
        service: string &log;
        duration: double &log;

        src_bytes: string &log;
        dst_bytes: string &log;
        conn_state: string &log;
        missed_bytes: string &log;
        src_pkts: string &log;
        src_ip_bytes: string &log;
        dst_pkts: string &log;
        dst_ip_bytes: string &log;

        ssl_version: string &log;
        ssl_cipher: string &log;
        ssl_resumed: string &log;
        ssl_established: string &log;

        weird_name: string &log;
        weird_addl: string &log;
        weird_notice: string &log;
    };
}

# SSL correlation table
type SSLExtra: record {
    version: string;
    cipher: string;
    resumed: string;
    established: string;
};

global ssl_map: table[string] of SSLExtra = table();

event SSL::log_ssl(rec: SSL::Info)
{
    ssl_map[rec$uid] = [
        $version     = rec?$version ? rec$version : "NaN",
        $cipher      = rec?$cipher ? rec$cipher : "NaN",
        $resumed     = rec?$resumed && rec$resumed ? "T" : "F",
        $established = rec?$established && rec$established ? "T" : "F"
    ];
}

# Main Event
event Conn::log_conn(rec: Conn::Info)
{
    # duration interval => double
    local duration_val: double = 0.0;
    if ( rec?$duration )
        duration_val = rec$duration / 1sec;

    # Converts transport_proto enum to "tcp", "udp", "icmp", ...
    local proto_val: string = "unknown";

    if ( rec$id$proto == 6 )
        proto_val = "tcp";
    else if ( rec$id$proto == 17 )
        proto_val = "udp";
    else if ( rec$id$proto == 1 )
        proto_val = "icmp";
    else
        proto_val = "protocol";

    # SSL defaults
    local ssl_version_val: string = "NaN";
    local ssl_cipher_val: string = "NaN";
    local ssl_resumed_val: string = "F";
    local ssl_established_val: string = "F";

    # Pull real SSL values if available
    if ( rec$uid in ssl_map ) {
        ssl_version_val     = ssl_map[rec$uid]$version;
        ssl_cipher_val      = ssl_map[rec$uid]$cipher;
        ssl_resumed_val     = ssl_map[rec$uid]$resumed;
        ssl_established_val = ssl_map[rec$uid]$established;
        delete ssl_map[rec$uid];
    }

    local info = RabbitMQ::ConnInfo(

        $src_port = fmt("%d", port_to_count(rec$id$orig_p)),
        $dst_port = fmt("%d", port_to_count(rec$id$resp_p)),
        $proto    = proto_val,
        $service  = rec?$service ? rec$service : "NaN",
        $duration = duration_val,

        $src_bytes    = rec?$orig_bytes ? fmt("%d", rec$orig_bytes) : "0",
        $dst_bytes    = rec?$resp_bytes ? fmt("%d", rec$resp_bytes) : "0",
        $conn_state   = rec$conn_state,
        $missed_bytes = rec?$missed_bytes ? fmt("%d", rec$missed_bytes) : "0",
        $src_pkts     = rec?$orig_pkts ? fmt("%d", rec$orig_pkts) : "0",
        $src_ip_bytes = rec?$orig_ip_bytes ? fmt("%d", rec$orig_ip_bytes) : "0",
        $dst_pkts     = rec?$resp_pkts ? fmt("%d", rec$resp_pkts) : "0",
        $dst_ip_bytes = rec?$resp_ip_bytes ? fmt("%d", rec$resp_ip_bytes) : "0",

        $ssl_version     = ssl_version_val,
        $ssl_cipher      = ssl_cipher_val,
        $ssl_resumed     = ssl_resumed_val,
        $ssl_established = ssl_established_val,

        $weird_name   = "NaN",
        $weird_addl   = "NaN",
        $weird_notice = "F"
    );

    print to_json(info);
}
