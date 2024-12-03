import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Function to convert rows into sentences
def convert_to_sentence(row):
    return (f"The connection lasted for {row['duration']} seconds, used the {row['protocol_type']} protocol, "
            f"and the service was {row['service']}. The flag was {row['flag']}, the source sent {row['src_bytes']} bytes, "
            f"and the destination sent {row['dst_bytes']} bytes. There were {row['num_failed_logins']} failed login attempts, "
            f"the user was {'logged in' if row['logged_in'] == 1 else 'not logged in'}, and the connection was labeled as {row['attack']}.")

# Columns for the dataset
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

# Load the training dataset
training = pd.read_csv('data/Train.txt', header=None)
training.columns = columns

# Encode attack labels into numeric values for SMOTE
label_encoder = LabelEncoder()
training['attack_encoded'] = label_encoder.fit_transform(training['attack'])

# Check class distribution and determine the smallest class size
class_counts = training['attack'].value_counts()
min_class_size = class_counts.min()

# Print class distribution for debugging
print("Class Distribution Before SMOTE:")
print(class_counts)

# SMOTE only works on numerical features, so we need to subset the numerical columns
numerical_columns = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

categorical_columns = ["protocol_type", "service", "flag", "land", "is_host_login", "is_guest_login", "logged_in"]

# Separate features and target
X = training[numerical_columns]
y = training['attack_encoded']

# Apply SMOTE with adjusted n_neighbors
smote = SMOTE(random_state=42, k_neighbors=min(min_class_size - 1, 5))
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print the class distribution after SMOTE
resampled_class_counts = pd.Series(y_resampled).value_counts()
print("\nClass Distribution After SMOTE:")
print(resampled_class_counts)

# Map resampled class indices back to their attack names
class_names = label_encoder.inverse_transform(resampled_class_counts.index)
for class_name, count in zip(class_names, resampled_class_counts):
    print(f"{class_name}: {count}")


# Create a DataFrame for resampled numerical features
resampled_data = pd.DataFrame(X_resampled, columns=numerical_columns)

# Map the categorical columns to the resampled data
# Match the number of resampled rows with the nearest neighbors from the original dataset
resampled_data[categorical_columns] = training.loc[smote.fit_resample(X, y)[1]].reset_index(drop=True)[categorical_columns]

# Add the resampled labels
resampled_data['attack_encoded'] = y_resampled
resampled_data['attack'] = label_encoder.inverse_transform(y_resampled)

# Convert the resampled dataset into sentences
resampled_data['text'] = resampled_data.apply(convert_to_sentence, axis=1)

# Save the new balanced training dataset
resampled_data[['text', 'attack']].to_csv('data/training_sentences.csv', index=False)

# Repeat the same process for the testing dataset if needed
testing = pd.read_csv('data/Test.txt', header=None)
testing.columns = columns
testing['text'] = testing.apply(convert_to_sentence, axis=1)
testing[['text', 'attack']].to_csv('data/testing_sentences.csv', index=False)

print("Balanced training dataset saved to 'data/training_sentences.csv'")
print("Testing dataset saved to 'data/testing_sentences.csv'")
