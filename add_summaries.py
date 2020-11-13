from py2neo import Graph
import wikipedia

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

people = graph.run("""MATCH (p:Person) WHERE NOT EXISTS (p.description) RETURN p.name AS name""").data()
for person in people:
    name = person['name']
    if name:
        try:
            search = wikipedia.search(name, results = 1, suggestion = True)
            summary = wikipedia.summary(search)
            if summary:
                summary = summary.replace('"', '')
                result = graph.run("""MATCH (p:Person) WHERE p.name="{}" SET p.description="{}" RETURN p.name AS name, p.description AS description""".format(name, summary))
                print(result)
        except:
            print("SEARCH & SUMMARY FAILED FOR: ", name, "- GIVING UP")
