import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random

# Initialize task list
tasks = pd.DataFrame(columns=['description', 'priority', 'due_date'])

# Load existing tasks
try:
    tasks = pd.read_csv('tasks.csv')
except FileNotFoundError:
    pass

# Save tasks to CSV
def save_tasks():
    tasks.to_csv('tasks.csv', index=False)

# Train ML model
vectorizer = CountVectorizer()
clf = MultinomialNB()
model = make_pipeline(vectorizer, clf)
if not tasks.empty:
    model.fit(tasks['description'], tasks['priority'])

# Add a task
def add_task(description, priority, due_date):
    global tasks
    new_task = pd.DataFrame({
        'description': [description],
        'priority': [priority],
        'due_date': [due_date]
    })
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()

# Remove a task
def remove_task(description):
    global tasks
    tasks = tasks[tasks['description'] != description]
    save_tasks()

# List all tasks
def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print(tasks)

# Recommend a task
def recommend_task():
    if not tasks.empty:
        high_priority_tasks = tasks[tasks['priority'] == 'High']
        if not high_priority_tasks.empty:
            random_task = random.choice(high_priority_tasks['description'].tolist())
            print(f"Recommended task: {random_task} - Priority: High")
        else:
            print("No high-priority tasks available for recommendation.")
    else:
        print("No tasks available for recommendations.")

# Filter tasks by priority
def filter_tasks_by_priority(priority_level):
    filtered = tasks[tasks['priority'].str.lower() == priority_level.lower()]
    if filtered.empty:
        print(f"No tasks found with priority '{priority_level}'.")
    else:
        print(filtered)

# Main menu
while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Tasks")
    print("4. Recommend Task")
    print("5. Filter Tasks by Priority")
    print("6. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ")
        priority = input("Enter task priority (Low/Medium/High): ").capitalize()
        due_date = input("Enter due date (YYYY-MM-DD): ")
        add_task(description, priority, due_date)
        print("✅ Task added successfully.")

    elif choice == "2":
        description = input("Enter task description to remove: ")
        remove_task(description)
        print("✅ Task removed successfully.")

    elif choice == "3":
        list_tasks()

    elif choice == "4":
        recommend_task()

    elif choice == "5":
        priority_level = input("Enter priority level to filter (Low/Medium/High): ")
        filter_tasks_by_priority(priority_level)

    elif choice == "6":
        print("👋 Goodbye!")
        break

    else:
        print("❌ Invalid option. Please select a valid option.")