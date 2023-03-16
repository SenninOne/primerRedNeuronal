import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

train_text = open("C:/Users/Braulio/Desktop/pythonProjects/primerRedNeuronal/train.txt").read()
train_encodings = tokenizer(train_text, return_tensors="pt")
input_ids = train_encodings.input_ids.to(device)
attention_mask = train_encodings.attention_mask.to(device)

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    model.train()
    outputs = model(input_ids, attention_mask=attention_mask)
    loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), input_ids.view(-1))
    loss.backward()
    optimizer.step()

model.eval()
prompt_text="El significado de la vida es"
prompt_encodings=tokenizer(prompt_text,return_tensors="pt")
input_ids=prompt_encodings.input_ids.to(device)
attention_mask=prompt_encodings.attention_mask.to(device)

sample_output=model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
)

print(tokenizer.decode(sample_output[0], skip_special_tokens=True))