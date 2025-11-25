# Introduction 
TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project. 

# Getting Started
TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

# Build and Test
TODO: Describe and show how to build your code and run the tests. 

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)

# Steps

# Install Docker Desktop

# Create .env file and get contents from Team

# To run Backend
1. docker compose up --build

# To configure/update tables in Database
1. docker compose exec server aerich init -t app.db.base.TORTOISE_ORM
2. docker compose exec server aerich upgrade

# In linux system, if permission denied
1. sudo docker compose exec --user root aerich init -t app.db.base.TORTOISE_ORM
2. sudo docker compose exec --user root server aerich upgrade

# To downgrade/rollback to previous state in Database
1. docker compose exec server aerich downgrade

# To create new Migration files/tables and upgrade in Database
1. docker compose exec server aerich migrate
2. docker compose exec server aerich upgrade

# In linux system, if permission denied
1. sudo docker compose exec --user root server aerich migrate
2. sudo docker compose exec --user root server aerich upgrade

# Connect to database
1. docker compose exec db psql -U postgres
2. \c <database_name>

# List tables
1. \dt

# To check celery tasks running in background
1. Open browser, Go to http://localhost:5556/workers

# For db and migrations issues
1. docker compose exec db psql -U postgres
2. DROP DATABASE example;
3. docker compose down -v
4. docker compose up -d
5. docker compose exec server aerich init -t app.db.base.TORTOISE_ORM
#apply existing migrations before creating new ones
6. docker compose exec server aerich upgrade

# To check app logs
1. az login --use-device-code
2. az account set --subscription <subscription_id>
3. az aks get-credentials --resource-group <rg_name> --name <aks_name> --overwrite-existing
4. kubectl get pods -n <namespace>
5. kubectl logs -f <pod_name> -n <namespace>