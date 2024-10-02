import os

# Ask the user for the remote repository URL
remote_url = input("Enter the remote repository URL: ")

# Run the Git commands using os.system
os.system('git init')
os.system('git add .')
os.system('git commit -m "first commit"')
os.system('git branch -M main')
os.system(f'git remote add origin {remote_url}')
os.system('git push -u origin main')

print("Repository initialized and pushed successfully.")
