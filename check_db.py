from pprint import pprint

from server.v2.repository import V2Repository
from server.v2.validation import ModelInvariantValidator


repository = V2Repository()
with repository.transaction() as connection:
    pprint(repository.stats(connection))
    pprint([dict(row) for row in connection.execute("SELECT * FROM scenes ORDER BY cloud_id")])
    pprint([dict(row) for row in connection.execute("SELECT * FROM word_forms ORDER BY cloud_id")])
pprint(ModelInvariantValidator(repository).validate())
