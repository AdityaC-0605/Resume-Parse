import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('AI_Resume_Screening.csv')

# Let's first explore the dataset
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nColumns:", df.columns.tolist())
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Analyze the distribution of job roles
print("\nJob role distribution:")
print(df['Job Role'].value_counts())

# Analyze the distribution of education levels
print("\nEducation distribution:")
print(df['Education'].value_counts())

# Analyze the distribution of certifications
print("\nCertifications distribution:")
print(df['Certifications'].value_counts())

# Analyze the distribution of recruiter decisions
print("\nRecruiter decision distribution:")
print(df['Recruiter Decision'].value_counts())

# Analyze the distribution of experience
plt.figure(figsize=(10, 6))
sns.histplot(df['Experience (Years)'], bins=10, kde=True)
plt.title('Distribution of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Frequency')
plt.savefig('experience_distribution.png')

# Analyze the relationship between AI Score and Recruiter Decision
plt.figure(figsize=(10, 6))
sns.boxplot(x='Recruiter Decision', y='AI Score (0-100)', data=df)
plt.title('AI Score vs Recruiter Decision')
plt.savefig('ai_score_vs_decision.png')

# Let's create a BERT-based model for resume parsing
# First, preprocess the data for BERT

# Create a text representation of each resume by combining relevant fields
df['Resume_Text'] = df.apply(
    lambda row: f"Name: {row['Name']} Skills: {row['Skills']} Experience: {row['Experience (Years)']} years "
    f"Education: {row['Education']} Certifications: {row['Certifications']} "
    f"Projects: {row['Projects Count']} Salary: {row['Salary Expectation ($)']}",
    axis=1
)

# Define job roles as the target classes
job_roles = df['Job Role'].unique().tolist()
role_to_id = {role: idx for idx, role in enumerate(job_roles)}
id_to_role = {idx: role for idx, role in enumerate(job_roles)}

# Tokenize the text for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class ResumeDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create labels
labels = df['Job Role'].map(role_to_id).values

# Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Resume_Text'].values, labels, test_size=0.2, random_state=42
)

# Create datasets
train_dataset = ResumeDataset(train_texts, train_labels)
test_dataset = ResumeDataset(test_texts, test_labels)

# Create data loaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define BERT-based model for resume classification
class ResumeClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResumeClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResumeClassifier(len(job_roles)).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Training function
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

# Evaluation function
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Calculate accuracy
    accuracy = sum(1 for x, y in zip(all_preds, all_labels) if x == y) / len(all_labels)
    
    return total_loss / len(data_loader), accuracy

# Train the model
num_epochs = 5
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("-" * 50)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('training_loss.png')

# Plot validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), test_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.savefig('validation_accuracy.png')

# Function to extract skills from a resume
def extract_skills(resume_text, model, tokenizer, device):
    # Define common skills to look for (based on your dataset)
    common_skills = [
        "Python", "SQL", "Java", "C++", "React", "TensorFlow", "PyTorch", 
        "NLP", "Deep Learning", "Machine Learning", "Cybersecurity", 
        "Networking", "Linux", "Ethical Hacking"
    ]
    
    # Find skills in the text
    found_skills = [skill for skill in common_skills if skill.lower() in resume_text.lower()]
    
    # Also use BERT to classify the job role
    encoding = tokenizer(
        resume_text,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
        _, preds = torch.max(outputs, dim=1)
    
    predicted_role = id_to_role[preds.item()]
    
    return {
        "skills": found_skills,
        "predicted_role": predicted_role
    }

# Function to parse a complete resume
def parse_resume(resume_text, model, tokenizer, device):
    # Extract name using a simple rule (assuming "Name: " prefix)
    name_match = resume_text.split("Name: ")[1].split(" Skills:")[0] if "Name: " in resume_text else "Unknown"
    
    # Extract skills and role using BERT
    skills_and_role = extract_skills(resume_text, model, tokenizer, device)
    
    # Extract experience using a simple rule (assuming "Experience: " prefix)
    experience_match = resume_text.split("Experience: ")[1].split(" years")[0] if "Experience: " in resume_text else "Unknown"
    
    # Extract education using a simple rule (assuming "Education: " prefix)
    education_match = resume_text.split("Education: ")[1].split(" Certifications:")[0] if "Education: " in resume_text else "Unknown"
    
    # Extract certifications using a simple rule (assuming "Certifications: " prefix)
    certifications_match = resume_text.split("Certifications: ")[1].split(" Projects:")[0] if "Certifications: " in resume_text else "None"
    
    return {
        "name": name_match.strip(),
        "skills": skills_and_role["skills"],
        "experience_years": experience_match.strip(),
        "education": education_match.strip(),
        "certifications": certifications_match.strip(),
        "predicted_role": skills_and_role["predicted_role"]
    }

# Test the resume parser on a few examples
print("\nTesting the resume parser on a few examples:")
for i in range(3):
    sample_resume = test_texts[i]
    parsed_result = parse_resume(sample_resume, model, tokenizer, device)
    
    print(f"\nExample {i+1}:")
    print("Resume text:", sample_resume[:100] + "...")
    print("Parsed result:", parsed_result)
    print("Actual job role:", id_to_role[test_labels[i]])

# Create a more advanced BERT-based information extraction model
# (In a real-world scenario, you would train named entity recognition models)

# Function to create a complete resume parsing system
def create_resume_parser():
    # Load the pre-trained model
    parser_model = ResumeClassifier(len(job_roles)).to(device)
    # In a real implementation, you would load the saved model weights
    # parser_model.load_state_dict(torch.load('resume_parser_model.pth'))
    
    def parse(resume_text):
        return parse_resume(resume_text, parser_model, tokenizer, device)
    
    return parse

# Create the parser
resume_parser = create_resume_parser()

# Example usage
print("\nExample usage of the resume parser:")
sample_resume = """
Name: John Doe Skills: Python, Machine Learning, TensorFlow, Data Analysis
Experience: 5 years Education: M.Tech Certifications: AWS Certified
Projects: 7 Salary: 95000
"""

parsed_data = resume_parser(sample_resume)
print("Parsed data:", parsed_data)

# Save the model for future use
# torch.save(model.state_dict(), 'resume_parser_model.pth')

print("\nResume parser implementation complete!")