import pandas as pd

def convert_to_sentence(row):
    return (f"The connection lasted for {row['duration']} seconds, used the {row['protocol_type']} protocol, "
            f"and the service was {row['service']}. The flag was {row['flag']}, the source sent {row['src_bytes']} bytes, "
            f"and the destination sent {row['dst_bytes']} bytes. There were {row['num_failed_logins']} failed login attempts, "
            f"the user was {'logged in' if row['logged_in'] == 1 else 'not logged in'}, and the connection was labeled as {row['attack']}.")

training = pd.read_csv('data/Train.txt')
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"
]
training.columns = columns

training['text'] = training.apply(convert_to_sentence, axis=1)
training[['text', 'attack']].to_csv('data/training_sentences.csv', index=False)

testing = pd.read_csv('data/Test.txt')
testing.columns = columns
testing['text'] = testing.apply(convert_to_sentence, axis=1)
testing[['text', 'attack']].to_csv('data/testing_sentences.csv', index=False)