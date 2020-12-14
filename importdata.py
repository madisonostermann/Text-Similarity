# Create Neo4j Database
# Install APOC & GDSL Plug-ins
# Change Settings
    # dbms.memory.heap.initial_size=1G
    # dbms.memory.heap.max_size=3G
    # apoc.import.file.enabled=true AFTER dbms.security.procedures.unrestricted=apoc.*,gds.* (will get errors otherwise)
    # dbms.security.allow_csv_import_from_file_urls=true
# Start up database

# Manually add CSV files to database's 'import' folder

# Create 'python' folder under database folder (using 3.7.4 in my environment)
    # Add provided requirements.txt and database_update_script.py (this file) to python folder
    # Set up a virtualenv in this python folder
        # pip install virtualenv
        # virtualenv --version (just to check)
        # virtualenv venv
        # source venv/bin/activate
        # pip install -r requirements.txt
# This file should be in the python folder

from py2neo import Graph

print("Neo4j DB Port: ")
port = input()
print("Neo4j DB Username: ")
user = input()
print("Neo4j DB Password: ")
pswd = input()

# Make sure the database is started first, otherwise attempt to connect will fail
try:
    graph = Graph('bolt://localhost:'+port, auth=(user, pswd))
    print('SUCCESS: Connected to the Neo4j Database.')
except Exception as e:
    print('ERROR: Could not connect to the Neo4j Database. See console for details.')
    raise SystemExit(e)

########################################################

query_list = []

#create person nodes
query_list.append("""CALL apoc.periodic.iterate("
LOAD CSV WITH HEADERS 
FROM 'file:///womeninstem.csv' AS line
RETURN line
","
MERGE (person:Person { name: line.personLabel })
ON CREATE SET person.birth = line.birthdate, person.death = line.deathdate
",{batchSize:1000, parallel:false, retries: 10})
YIELD operations""")

#create occupation nodes
query_list.append("""CALL apoc.periodic.iterate("
LOAD CSV WITH HEADERS 
FROM 'file:///womeninstem.csv' AS line
RETURN line
","
MERGE (occ:Occupation { title: line.occupationLabel })
",{batchSize:1000, parallel:false, retries: 10})
YIELD operations""")

#create nation nodes
query_list.append("""CALL apoc.periodic.iterate("
LOAD CSV WITH HEADERS 
FROM 'file:///womeninstem.csv' AS line
RETURN line
","
MERGE (nat:Nation { title: coalesce(line.nationalityLabel, ' ') })
",{batchSize:1000, parallel:false, retries: 10})
YIELD operations""")

#create award nodes
query_list.append("""CALL apoc.periodic.iterate("
LOAD CSV WITH HEADERS 
FROM 'file:///womeninstem.csv' AS line
RETURN line
","
MERGE (award:Award { title: coalesce(line.awardLabel, ' ') })
",{batchSize:1000, parallel:false, retries: 10})
YIELD operations""")

#relate person and occupation nodes
query_list.append("""CALL apoc.periodic.iterate("
LOAD CSV WITH HEADERS 
FROM 'file:///womeninstem.csv' AS line
RETURN line
","
MATCH (person:Person { name: line.personLabel })
MATCH (occ:Occupation { title: line.occupationLabel })
MERGE (person)-[:works_as]->(occ)
",{batchSize:1000, parallel:false, retries: 10})
YIELD operations""")

#relate person and nation nodes 
query_list.append("""CALL apoc.periodic.iterate("
LOAD CSV WITH HEADERS 
FROM 'file:///womeninstem.csv' AS line
RETURN line
","
MATCH (person:Person { name: line.personLabel })
MATCH (nat:Nation { title: line.nationalityLabel })
MERGE (person)-[:citizen_of]->(nat)
",{batchSize:1000, parallel:false, retries: 10})
YIELD operations""")

#relate person and award nodes
query_list.append("""CALL apoc.periodic.iterate("
LOAD CSV WITH HEADERS 
FROM 'file:///womeninstem.csv' AS line
RETURN line
","
MATCH (person:Person { name: line.personLabel })
MATCH (award:Award { title: line.awardLabel })
MERGE (award)-[:awarded_to]->(person)
",{batchSize:1000, parallel:false, retries: 10})
YIELD operations""")

#create more person nodes
query_list.append("""CALL apoc.periodic.iterate("
LOAD CSV WITH HEADERS 
FROM 'file:///institutions.csv' AS line
RETURN line
","
MERGE (person:Person { name: line.personLabel })
",{batchSize:1000, parallel:false, retries: 10})
YIELD operations""")

#create institution nodes
query_list.append("""CALL apoc.periodic.iterate("
LOAD CSV WITH HEADERS 
FROM 'file:///institutions.csv' AS line
RETURN line
","
MERGE (i:Institution { title: coalesce(line.institutionLabel, ' ') })
",{batchSize:1000, parallel:false, retries: 10})
YIELD operations""")

#relate person and institution nodes
query_list.append("""CALL apoc.periodic.iterate("
LOAD CSV WITH HEADERS 
FROM 'file:///institutions.csv' AS line
RETURN line
","
MATCH (person:Person { name: line.personLabel })
MATCH (i:Institution { title: line.institutionLabel })
MERGE (i)<-[:educated_at]->(person)
",{batchSize:1000, parallel:false, retries: 10})
YIELD operations""")

for query in query_list:        
   g = graph.begin() # open transaction        
   result = g.run(query).to_data_frame() # execute query
   try:            
      if result['operations'][0]['failed'] > 0: #means batch failed
         print('Could not finish query')
      else: #means batch succeeded                
         g.commit() # close transaction
         print('Completed query ')
   except Exception as e: #means there's a syntax or other error         
      g.commit() # close transaction  
      print('Completed query')


#remove blank nodes
graph.run("""MATCH (n) WHERE n.title = ' ' DETACH DELETE n""")

#remove duplicates
graph.run("""MATCH (n:Person)
WITH n.name AS name, collect(n) AS nodes
WHERE size(nodes) > 1
FOREACH (n in tail(nodes) | DETACH DELETE n)""")

#remove people who came up with numbers in name
graph.run("""MATCH (p:Person)
WHERE p.name CONTAINS "1" 
DETACH DELETE p""")
graph.run("""MATCH (p:Person)
WHERE p.name CONTAINS "2" 
DETACH DELETE p""")
graph.run("""MATCH (p:Person)
WHERE p.name CONTAINS "3" 
DETACH DELETE p""")
graph.run("""MATCH (p:Person)
WHERE p.name CONTAINS "4" 
DETACH DELETE p""")
graph.run("""MATCH (p:Person)
WHERE p.name CONTAINS "5" 
DETACH DELETE p""")
graph.run("""MATCH (p:Person)
WHERE p.name CONTAINS "6" 
DETACH DELETE p""")
graph.run("""MATCH (p:Person)
WHERE p.name CONTAINS "7" 
DETACH DELETE p""")
graph.run("""MATCH (p:Person)
WHERE p.name CONTAINS "8" 
DETACH DELETE p""")
graph.run("""MATCH (p:Person)
WHERE p.name CONTAINS "9" 
DETACH DELETE p""")
