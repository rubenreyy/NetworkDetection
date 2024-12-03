from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np

# Tokenize the data
model_name = 'bert-base-uncased'
dataset = load_dataset('csv', data_files={'train': 'data/training_sentences.csv', 'test': 'data/testing_sentences.csv'})

# Explicitly access the splits
train_dataset = dataset['train']
test_dataset = dataset['test']

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenizing(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

# Tokenize training and testing datasets
tokenized_training = train_dataset.map(tokenizing, batched=True)
tokenized_testing = test_dataset.map(tokenizing, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Label mapping
label_mapping = {
    "normal": 0,  # Normal

    # DOS (Denial of Service)
    "back": 1,
    "land": 1,
    "neptune": 1,
    "pod": 1,
    "smurf": 1,
    "teardrop": 1,
    "apache2": 1,
    "mailbomb": 1,
    "processtable": 1,
    "udpstorm": 1,

    # R2L (Remote to Local)
    "ftp_write": 2,
    "guess_passwd": 2,
    "imap": 2,
    "multihop": 2,
    "phf": 2,
    "warezmaster": 2,
    "warezclient": 2,
    "snmpguess": 2,
    "httptunnel": 2,
    "sendmail": 2,
    "named": 2,
    "xsnoop": 2,
    "worm": 2,
    "spy": 2,

    # U2R (User to Root)
    "buffer_overflow": 3,
    "loadmodule": 3,
    "perl": 3,
    "rootkit": 3,
    "xterm": 3,
    "ps": 3,
    "xlock": 3,
    "sqlattack": 3,

    # Probe
    "ipsweep": 4,
    "nmap": 4,
    "portsweep": 4,
    "satan": 4,
    "saint": 4,
    "mscan": 4,
    "snmpgetattack": 4
}

# Map attack types to categories (labels)
def map_labels(example):
    return {"labels": label_mapping.get(example["attack"], -1)}

train_dataset = train_dataset.map(map_labels)
test_dataset = test_dataset.map(map_labels)

# Ensure labels are included in the tokenized datasets
tokenized_training = tokenized_training.map(lambda x: {"labels": label_mapping[x["attack"]]})
tokenized_testing = tokenized_testing.map(lambda x: {"labels": label_mapping[x["attack"]]})

# Calculate class weights
labels = tokenized_training['labels']
class_counts = np.bincount(labels)  # Count occurrences of each class label
class_weights = 1.0 / class_counts
normalized_weights = class_weights / class_weights.sum()  # Normalize to ensure weights add to 1
weights = torch.tensor(normalized_weights).float().to('cuda')  # Move weights to GPU

print(f"Class Counts: {class_counts}")
print(f"Class Weights: {weights}")

# Override the Trainer to include weighted loss
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.1,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
)

# Initialize Trainer with weighted loss
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_training,
    eval_dataset=tokenized_testing,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Save the model
save_directory = "./fine_tuned_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# Predict and generate the classification report
predictions, labels, _ = trainer.predict(tokenized_testing)
predicted_classes = predictions.argmax(axis=1)
print(classification_report(labels, predicted_classes))

# Confusion Matrix
cm = confusion_matrix(labels, predicted_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
